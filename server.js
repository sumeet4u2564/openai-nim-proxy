// server.js - OpenAI to NVIDIA NIM API Proxy (improved)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'x-api-key', 'anthropic-version'],
  exposedHeaders: ['Content-Type'],
  credentials: false
}));
app.options('*', cors());
app.use(express.json({ limit: '10mb' }));

// ─── Config ──────────────────────────────────────────────────────────────────
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// 🔥 CONTROLS
const SHOW_REASONING   = false; // true = wrap model thinking in <think> tags
const INJECT_FORMAT    = true;  // true = inject formatting instructions into system prompt
                                //        (reduces wall-of-text, improves RP quality)

// Per-model thinking mode: ONLY enable for models that actually support & benefit from it.
// Enabling this on non-reasoning models wastes tokens and slows everything down.
const ENABLE_THINKING_MODE = true; // Set to false to disable for all models

const THINKING_MODELS = new Set(
  ENABLE_THINKING_MODE ? Object.values(MODEL_MAPPING) : []
);

// ─── Model mapping ────────────────────────────────────────────────────────────
// Rules of thumb:
//   • gpt-3.5 / small slots → fast 8-70B models
//   • gpt-4 / large slots   → 70B-405B or specialist models
//   • Avoid *-thinking variants for roleplay unless you WANT slow CoT reasoning
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',          // fast, good instruction follower
  'gpt-4':         'nvidia/llama-3.1-nemotron-ultra-253b-v1', // strong reasoning + roleplay
  'gpt-4-turbo':   'meta/llama-3.1-405b-instruct',          // large but reliable
  'gpt-4o':        'deepseek-ai/deepseek-v3.1',             // very capable, no thinking overhead
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':    'qwen/qwen3-next-80b-a3b-thinking',      // kept but thinking is gated below
};

// Models that tend to ignore system prompts → get extra enforcement
const STUBBORN_MODEL_PREFIXES = ['qwen/', 'deepseek-'];

// ─── Formatting injection ─────────────────────────────────────────────────────
// Appended to system prompt when INJECT_FORMAT = true.
// Tune this to your Janitor AI use-case. The key rules here are:
//   1. Short paragraphs kill wall-of-text.
//   2. Explicit ban on lists/markdown prevents asterisk spam.
//   3. Roleplay-specific rules keep the model in character.
const FORMAT_INJECTION = `

[Response Formatting Rules — follow these exactly]
- Write in paragraphs. Never output a wall of text.
- Do NOT use bullet points, numbered lists, or markdown headers.
- Do NOT use asterisks for actions (*smiles*, *nods*) — write actions in plain prose instead.
- Stay in character at all times. Do not break the fourth wall or explain your reasoning.
- Match the tone and vocabulary established in the system prompt above.
- Write longer responses, minimum 400 words per replay.
- If continuing a scene, pick up exactly where the last message left off without recap.`;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Merge duplicate system messages into one block */
function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return messages;
  const sys    = messages.filter(m => m.role === 'system');
  const nonSys = messages.filter(m => m.role !== 'system');
  if (sys.length <= 1) return messages;
  return [{ role: 'system', content: sys.map(m => m.content).join('\n\n') }, ...nonSys];
}

/** Inject formatting rules into the system prompt */
function injectFormatting(messages) {
  if (!INJECT_FORMAT) return messages;
  return messages.map(m =>
    m.role === 'system'
      ? { ...m, content: m.content + FORMAT_INJECTION }
      : m
  );
}

/** For models that tend to ignore system prompts, reinforce them via user turn */
function enforceSystemPrompt(messages, nimModel) {
  const isStubborn = STUBBORN_MODEL_PREFIXES.some(p => nimModel.includes(p));
  if (!isStubborn) return messages;

  const systemMsg = messages.find(m => m.role === 'system');
  if (!systemMsg) return messages;

  const others = messages.filter(m => m.role !== 'system');
  return [
    systemMsg,
    ...others.map((m, i) =>
      i === 0 && m.role === 'user'
        ? { ...m, content: `[IMPORTANT: Follow the system prompt above strictly and completely]\n\n${m.content}` }
        : m
    )
  ];
}

/** Build the NIM request body */
function buildNimRequest(body, nimModel) {
  const { messages, temperature, max_tokens, stream, top_p, stop, frequency_penalty, presence_penalty } = body;

  const useThinking = THINKING_MODELS.has(nimModel);

  const processedMessages = enforceSystemPrompt(
    injectFormatting(
      normalizeMessages(messages)
    ),
    nimModel
  );

  return {
    model: nimModel,
    messages: processedMessages,
    temperature:  temperature ?? 1.0,
    max_tokens:   max_tokens || 3024,  // ⬇️ reduced from 9024 — big budgets cause rambling
    ...(top_p              !== undefined && { top_p }),
    ...(stop               !== undefined && { stop }),
    ...(frequency_penalty  !== undefined && { frequency_penalty }),
    ...(presence_penalty   !== undefined && { presence_penalty }),
    // Only send thinking param to models that actually support it
    ...(useThinking && { extra_body: { chat_template_kwargs: { thinking: true } } }),
    stream: stream ?? true,
  };
}

// ─── Health check ─────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    format_injection:  INJECT_FORMAT,
    thinking_models:   [...THINKING_MODELS],
  });
});

// ─── List models ──────────────────────────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// ─── Chat completions ─────────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, stream } = req.body;

    // ── Model selection with size-based fallback ──
    let nimModel = MODEL_MAPPING[model];

    if (!nimModel) {
      // Try the name directly against NIM
      try {
        const testRes = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { model, messages: [{ role: 'user', content: 'hi' }], max_tokens: 1 },
          {
            headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            validateStatus: s => s < 500,
          }
        );
        if (testRes.status >= 200 && testRes.status < 300) nimModel = model;
      } catch (_) {}

      if (!nimModel) {
        const lower = model.toLowerCase();
        if      (lower.includes('gpt-4') || lower.includes('opus') || lower.includes('405b')) nimModel = 'meta/llama-3.1-405b-instruct';
        else if (lower.includes('claude') || lower.includes('gemini') || lower.includes('70b')) nimModel = 'meta/llama-3.3-70b-instruct';
        else    nimModel = 'meta/llama-3.1-8b-instruct';
      }
    }

    const nimRequest = buildNimRequest(req.body, nimModel);

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      // ⬇️ Generous timeout; NIM can be slow to start large models
      timeout: 120_000,
    });

    // ── Streaming ──
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer          = '';
      let reasoningBuffer = ''; // accumulate reasoning across chunks
      let reasoningOpen   = false;

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            // Close any dangling <think> block before finishing
            if (reasoningOpen && SHOW_REASONING) {
              const closeChunk = {
                id: `chatcmpl-close-${Date.now()}`,
                object: 'chat.completion.chunk',
                choices: [{ index: 0, delta: { content: '\n</think>\n\n' }, finish_reason: null }]
              };
              res.write(`data: ${JSON.stringify(closeChunk)}\n\n`);
              reasoningOpen = false;
            }
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content ?? null;
              const content   = delta.content ?? null;
              delete delta.reasoning_content; // always strip from wire format

              if (SHOW_REASONING) {
                // ── Open <think> on first reasoning chunk ──
                if (reasoning !== null && !reasoningOpen) {
                  delta.content = '<think>\n' + reasoning;
                  reasoningOpen = true;
                } else if (reasoning !== null) {
                  delta.content = reasoning;
                }

                // ── Close </think> when real content arrives ──
                if (content !== null && reasoningOpen) {
                  delta.content = (delta.content ?? '') + '\n</think>\n\n' + content;
                  reasoningOpen = false;
                } else if (content !== null) {
                  delta.content = (delta.content ?? '') + content;
                }

                // If delta.content is still undefined (reasoning=null, content=null), set ''
                if (delta.content === undefined) delta.content = '';
              } else {
                // Strip reasoning entirely; only forward real content
                delta.content = content ?? '';
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (_) {
            // Unparseable line — forward as-is (e.g. comment lines)
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end',   ()    => res.end());
      response.data.on('error', (err) => { console.error('Stream error:', err.message); res.end(); });

    } else {
      // ── Non-streaming ──
      const choices = response.data.choices.map(choice => {
        let fullContent = choice.message?.content ?? '';
        if (SHOW_REASONING && choice.message?.reasoning_content) {
          fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
        }
        return {
          index:         choice.index,
          message:       { role: choice.message.role, content: fullContent },
          finish_reason: choice.finish_reason,
        };
      });

      res.json({
        id:      `chatcmpl-${Date.now()}`,
        object:  'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices,
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    const status  = error.response?.status ?? 500;
    const message = error.response?.data?.error?.message ?? error.message ?? 'Internal server error';
    console.error(`Proxy error [${status}]:`, message);
    res.status(status).json({ error: { message, type: 'invalid_request_error', code: status } });
  }
});

// ─── 404 catch-all ────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, type: 'invalid_request_error', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\nOpenAI → NVIDIA NIM Proxy  |  port ${PORT}`);
  console.log(`  Reasoning display : ${SHOW_REASONING  ? 'ON' : 'OFF'}`);
  console.log(`  Format injection  : ${INJECT_FORMAT   ? 'ON' : 'OFF'}`);
  console.log(`  Thinking models   : ${[...THINKING_MODELS].join(', ')}\n`);
});
