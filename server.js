// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 MINIMUM RESPONSE TOKENS - AI must produce at least this many tokens
const MIN_TOKENS = 400; // Set to 0 to disable

// 🔥 CUSTOM SYSTEM PROMPT - Write your instructions here (e.g. formatting, tone, style)
// Leave as empty string '' to disable
const CUSTOM_SYSTEM_PROMPT = `
Always write responses using proper paragraphs with clear spacing between them.
Never use bullet points or numbered lists unless explicitly asked.
Be descriptive and elaborate in your answers.
`.trim();

// 🔥 USER MESSAGE SUFFIX - Appended to the last user message every request.
// The model sees this right before it starts generating, so it's very hard to ignore.
// Edit the text inside the backticks to whatever formatting rules you want enforced.
// Set to empty string '' to disable.
const USER_MESSAGE_SUFFIX = `\n\n[Formatting rule: Write your response in clearly separated paragraphs. Press Enter twice between each paragraph. Never write a wall of unbroken text.], [Roleplay rule: The user's character belongs exclusively to the user. Do not write any dialogue, internal thoughts, physical actions, or decisions for them — not even small ones like nodding or sighing. React to the user, never speak for them.],`;

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'deepseek-ai/deepseek-v4-pro': 'deepseek-ai/deepseek-v4-flash',
};

// Helper: inject custom system prompt into message array
function injectSystemPrompt(messages) {
  if (!CUSTOM_SYSTEM_PROMPT) return messages;

  const systemIndex = messages.findIndex(m => m.role === 'system');
  if (systemIndex !== -1) {
    // Merge with existing system message
    const updated = [...messages];
    updated[systemIndex] = {
      ...updated[systemIndex],
      content: CUSTOM_SYSTEM_PROMPT + '\n\n' + updated[systemIndex].content
    };
    return updated;
  } else {
    // Prepend a fresh system message
    return [{ role: 'system', content: CUSTOM_SYSTEM_PROMPT }, ...messages];
  }
}

// Helper: append suffix to the last user message so the model sees it right before generating
function injectUserSuffix(messages) {
  if (!USER_MESSAGE_SUFFIX) return messages;

  const updated = [...messages];
  for (let i = updated.length - 1; i >= 0; i--) {
    if (updated[i].role === 'user') {
      updated[i] = {
        ...updated[i],
        content: updated[i].content + USER_MESSAGE_SUFFIX
      };
      break;
    }
  }
  return updated;
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    min_tokens: MIN_TOKENS,
    custom_system_prompt: CUSTOM_SYSTEM_PROMPT ? 'SET' : 'NOT SET',
    user_message_suffix: USER_MESSAGE_SUFFIX ? 'SET' : 'NOT SET'
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) nimModel = model;
        });
      } catch (e) {}

      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }

    // Inject custom system prompt + user message suffix
    const finalMessages = injectUserSuffix(injectSystemPrompt(messages));

    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: finalMessages,
      temperature: temperature || 0.8,
      max_tokens: max_tokens || 9024,
      min_tokens: MIN_TOKENS || undefined,   // 🔥 Minimum response length
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (line.startsWith('data: ')) {
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
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
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

// Catch-all for unsupported endpoints
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
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Min tokens: ${MIN_TOKENS}`);
  console.log(`Custom system prompt: ${CUSTOM_SYSTEM_PROMPT ? 'SET' : 'NOT SET'}`);
});
