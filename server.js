// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware — explicit CORS so browsers don't block preflight or streaming
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key', 'anthropic-version'],
  exposedHeaders: ['Content-Type'],
  credentials: false
}));

// Handle OPTIONS preflight explicitly (some clients send it before POST)
app.options('*', cors());

app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
// NOTE: Avoid *-thinking model variants if instruction-following is important
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-4': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Models known to sometimes ignore system prompts — gets an extra nudge
const STUBBORN_MODEL_PREFIXES = ['qwen/', 'deepseek-'];

// ─── Helper: merge multiple system messages into one ────────────────────────
function normalizeMessages(messages) {
  if (!messages || !Array.isArray(messages)) return messages;

  const systemMessages = messages.filter(m => m.role === 'system');
  const nonSystemMessages = messages.filter(m => m.role !== 'system');

  if (systemMessages.length <= 1) return messages;

  // Merge all system messages into a single one
  const mergedSystem = {
    role: 'system',
    content: systemMessages.map(m => m.content).join('\n\n')
  };

  return [mergedSystem, ...nonSystemMessages];
}

// ─── Helper: nudge stubborn models to follow system instructions ─────────────
function enforceSystemPrompt(messages, nimModel) {
  const hasSystem = messages.some(m => m.role === 'system');
  const isStubborn = STUBBORN_MODEL_PREFIXES.some(prefix => nimModel.includes(prefix));

  if (isStubborn && hasSystem) {
    const systemContent = messages.find(m => m.role === 'system')?.content;
    const otherMessages = messages.filter(m => m.role !== 'system');

    return [
      { role: 'system', content: systemContent },
      ...otherMessages.map((m, i) =>
        i === 0 && m.role === 'user'
          ? { ...m, content: `[Follow the system instructions strictly]\n\n${m.content}` }
          : m
      )
    ];
  }

  return messages;
}

// ─── Helper: prepare and clean the NIM request body ─────────────────────────
function buildNimRequest(body, nimModel) {
  const {
    messages,
    temperature,
    max_tokens,
    stream,
    top_p,
    stop,
    frequency_penalty,
    presence_penalty
  } = body;

  const processedMessages = enforceSystemPrompt(normalizeMessages(messages), nimModel);

  return {
    model: nimModel,
    messages: processedMessages,
    // Use ?? so that explicit 0 is respected (fixes the || 0.6 bug)
    temperature: temperature ?? 1.0,
    max_tokens: max_tokens || 9024,
    // Only include optional params if they were actually provided
    ...(top_p !== undefined && { top_p }),
    ...(stop !== undefined && { stop }),
    ...(frequency_penalty !== undefined && { frequency_penalty }),
    ...(presence_penalty !== undefined && { presence_penalty }),
    ...(ENABLE_THINKING_MODE && { extra_body: { chat_template_kwargs: { thinking: true } } }),
    stream: stream || true
  };
}

// ─── Health check ────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// ─── List models (OpenAI compatible) ────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({ object: 'list', data: models });
});

// ─── Chat completions (main proxy) ──────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, stream } = req.body;

    // ── Smart model selection with fallback ──
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Try the model name directly against the NIM API
      try {
        const testRes = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { model, messages: [{ role: 'user', content: 'test' }], max_tokens: 1 },
          {
            headers: {
              Authorization: `Bearer ${NIM_API_KEY}`,
              'Content-Type': 'application/json'
            },
            validateStatus: s => s < 500
          }
        );
        if (testRes.status >= 200 && testRes.status < 300) {
          nimModel = model;
        }
      } catch (_) {}

      // Size-based fallback
      if (!nimModel) {
        const lower = model.toLowerCase();
        if (lower.includes('gpt-4') || lower.includes('claude-opus') || lower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    const nimRequest = buildNimRequest(req.body, nimModel);

    // ── Make request to NVIDIA NIM ──
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // ── Streaming response ──
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('X-Accel-Buffering', 'no'); // Disables proxy buffering (nginx etc.)

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write(line + '\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (data.choices?.[0]?.delta) {
              const reasoning = data.choices[0].delta.reasoning_content;
              const content = data.choices[0].delta.content;

              if (SHOW_REASONING) {
                let combinedContent = '';

                if (reasoning && !reasoningStarted) {
                  combinedContent = '<think>\n' + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combinedContent = reasoning;
                }

                if (content && reasoningStarted) {
                  combinedContent += '</think>\n\n' + content;
                  reasoningStarted = false;
                } else if (content) {
                  combinedContent += content;
                }

                if (combinedContent) {
                  data.choices[0].delta.content = combinedContent;
                }
              } else {
                // Strip reasoning, pass only real content
                data.choices[0].delta.content = content || '';
              }

              delete data.choices[0].delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            res.write(line + '\n');
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // ── Non-streaming response ──
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent =
              '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }

          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);

    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// ─── Catch-all for unsupported endpoints ────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode:     ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
});
