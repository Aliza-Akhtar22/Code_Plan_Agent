import { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! Upload a CSV file to begin.", sender: 'bot' }
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Upload-related
  const [datasetId, setDatasetId] = useState(null);
  const [fileName, setFileName] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [chatError, setChatError] = useState('');

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addBotMessage = (text, isError = false) => {
    setMessages(prev => [
      ...prev,
      { id: Date.now() + Math.random(), text, sender: 'bot', isError }
    ]);
  };

  // Bot message that can include a table preview
  const addBotPreviewMessage = ({ filename, dataset_id, headRows }) => {
    setMessages(prev => [
      ...prev,
      {
        id: Date.now() + Math.random(),
        sender: 'bot',
        isError: false,
        kind: 'upload_preview',
        filename,
        dataset_id,
        headRows: Array.isArray(headRows) ? headRows : [],
      }
    ]);
  };

  // Forecast message that renders summary + tables
  const addBotForecastMessage = ({ assistantText, results, isError }) => {
    setMessages(prev => [
      ...prev,
      {
        id: Date.now() + Math.random(),
        sender: 'bot',
        isError: Boolean(isError),
        kind: 'forecast',
        assistantText: assistantText || '',
        results: results || null,
      }
    ]);
  };

  const addUserMessage = (text) => {
    setMessages(prev => [
      ...prev,
      { id: Date.now() + Math.random(), text, sender: 'user' }
    ]);
  };

  // ✅ Strip the raw "Forecast (head/tail)" blocks from assistantText
  const cleanAssistantText = (text) => {
    if (!text) return '';
    let t = String(text);

    // Cut off everything from "Forecast (head)" onwards (covers your current backend format)
    const cutMarkers = [
      "Forecast (head):",
      "Forecast(head):",
      "Forecast head:",
      "Forecast (tail):",
      "Forecast(tail):",
      "Forecast tail:",
      "\nResults:",
      "Results:"
    ];

    let idx = -1;
    for (const m of cutMarkers) {
      const i = t.indexOf(m);
      if (i !== -1) {
        idx = (idx === -1) ? i : Math.min(idx, i);
      }
    }

    if (idx !== -1) {
      t = t.slice(0, idx);
    }

    // Clean extra whitespace
    t = t.trim();

    // If it becomes empty, provide a minimal header
    if (!t) return "Forecast completed.";
    return t;
  };

  // Generic table renderer for arrays of objects
  const DataTable = ({ rows, title }) => {
    if (!rows || !Array.isArray(rows) || rows.length === 0) return null;

    const colSet = new Set();
    rows.forEach(r => Object.keys(r || {}).forEach(k => colSet.add(k)));
    const columns = Array.from(colSet);

    return (
      <div style={{ marginTop: '0.75rem', overflowX: 'auto' }}>
        {title && (
          <div style={{ fontWeight: 600, marginBottom: '0.5rem', color: 'rgba(255,255,255,0.95)' }}>
            {title}
          </div>
        )}
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  style={{
                    textAlign: 'left',
                    padding: '0.5rem',
                    borderBottom: '1px solid rgba(255,255,255,0.15)',
                    color: 'rgba(255,255,255,0.9)',
                    fontWeight: 600,
                    fontSize: '0.9rem',
                    whiteSpace: 'nowrap'
                  }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              <tr key={idx}>
                {columns.map((c) => (
                  <td
                    key={c}
                    style={{
                      textAlign: 'left',
                      padding: '0.5rem',
                      borderBottom: '1px solid rgba(255,255,255,0.08)',
                      color: 'rgba(255,255,255,0.85)',
                      fontSize: '0.9rem',
                      verticalAlign: 'top',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    {r?.[c] === null || r?.[c] === undefined ? '' : String(r[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const ForecastSummary = ({ results }) => {
    if (!results) return null;

    const trainingRows = results.training_rows ?? results.trainingRows;
    const inputRows = results.input_rows ?? results.inputRows;
    const cfg = results.config_used || results.configUsed || null;

    const items = [
      trainingRows !== undefined ? { k: "Training rows", v: trainingRows } : null,
      inputRows !== undefined ? { k: "Input rows", v: inputRows } : null,
      cfg?.ds_col ? { k: "ds column", v: cfg.ds_col } : null,
      cfg?.y_col ? { k: "y column", v: cfg.y_col } : null,
      cfg?.freq ? { k: "Frequency", v: cfg.freq } : null,
      cfg?.periods ? { k: "Forecast periods", v: cfg.periods } : null,
      Array.isArray(cfg?.regressors) ? { k: "Regressors", v: cfg.regressors.length ? cfg.regressors.join(", ") : "None" } : null,
    ].filter(Boolean);

    if (items.length === 0) return null;

    return (
      <div
        style={{
          marginTop: '0.25rem',
          padding: '0.75rem',
          borderRadius: '12px',
          background: 'rgba(0,0,0,0.15)',
          border: '1px solid rgba(255,255,255,0.08)'
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: '0.5rem', color: 'rgba(255,255,255,0.95)' }}>
          Forecast Summary
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', rowGap: '0.35rem', columnGap: '0.75rem' }}>
          {items.map((it, idx) => (
            <div key={idx} style={{ display: 'contents' }}>
              <div style={{ color: 'rgba(255,255,255,0.75)', fontSize: '0.9rem' }}>{it.k}</div>
              <div style={{ color: 'rgba(255,255,255,0.92)', fontSize: '0.9rem', fontWeight: 600 }}>{it.v}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadError('');
    setChatError('');
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const data = await response.json();

      if (!data.ok) {
        throw new Error(data.error || 'Upload failed');
      }

      setDatasetId(data.dataset_id);
      setFileName(data.filename || file.name);

      addBotPreviewMessage({
        filename: data.filename || file.name,
        dataset_id: data.dataset_id,
        headRows: data.preview?.head || [],
      });

      addBotMessage(`Now type a message like "hi" to get the plan and proposed ds/y, then reply "confirm".`);

    } catch (err) {
      console.error(err);
      const msg = err?.message || 'Failed to upload file.';
      setUploadError(msg);
      addBotMessage(`Upload error: ${msg}`, true);
    } finally {
      setIsLoading(false);
      e.target.value = '';
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    setChatError('');

    if (!input.trim()) return;

    if (!datasetId) {
      addBotMessage("Please upload a CSV file first (above).", true);
      return;
    }

    const text = input.trim();
    addUserMessage(text);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          message: text,
          show_code: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`);
      }

      const data = await response.json();

      if (!data.ok) {
        throw new Error(data.error || 'Chat failed');
      }

      const assistantText =
        data.assistant_message ||
        data.reply ||
        data.message ||
        "";

      const results = data.results || null;
      const hasForecastTables =
        results &&
        (Array.isArray(results.forecast_head) || Array.isArray(results.forecast_tail));

      if (hasForecastTables) {
        addBotForecastMessage({
          assistantText,
          results,
          isError: Boolean(data.error),
        });
      } else {
        let botMessageText = assistantText;

        if (results) {
          botMessageText += `\n\nResults:\n${JSON.stringify(results, null, 2)}`;
        }

        if (data.error) {
          botMessageText += `\n\nError:\n${data.error}`;
        }

        if (!botMessageText.trim()) {
          botMessageText = JSON.stringify(data, null, 2);
        }

        addBotMessage(botMessageText, Boolean(data.error));
      }

    } catch (error) {
      console.error('Error:', error);
      const msg = error?.message || "Sorry, something went wrong. Please try again.";
      setChatError(msg);
      addBotMessage(`Error: ${msg}`, true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>AI Assistant</h2>

        <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            disabled={isLoading}
          />
          <div style={{ color: 'rgba(255,255,255,0.8)', fontSize: '0.9rem' }}>
            {datasetId
              ? `Loaded: ${fileName} (dataset_id set)`
              : 'No file uploaded yet'}
          </div>
        </div>

        {uploadError && <div className="error-message">{uploadError}</div>}
        {chatError && <div className="error-message">{chatError}</div>}
      </div>

      <div className="messages-area">
        {messages.map((msg) => {
          if (msg.kind === 'upload_preview') {
            return (
              <div
                key={msg.id}
                className="message bot"
                style={{ whiteSpace: 'pre-wrap' }}
              >
                <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                  File uploaded successfully: {msg.filename}
                </div>
                <div style={{ marginBottom: '0.5rem' }}>
                  Dataset ID: {msg.dataset_id}
                </div>
                <div style={{ fontWeight: 600, marginTop: '0.75rem' }}>
                  Top 5 rows:
                </div>
                <DataTable rows={msg.headRows} />
              </div>
            );
          }

          // ✅ Forecast: show ONLY clean header + summary + tables
          if (msg.kind === 'forecast') {
            const results = msg.results || {};
            const head = Array.isArray(results.forecast_head) ? results.forecast_head : [];
            const tail = Array.isArray(results.forecast_tail) ? results.forecast_tail : [];

            const headerText = cleanAssistantText(msg.assistantText);

            return (
              <div
                key={msg.id}
                className={`message bot ${msg.isError ? 'error' : ''}`}
              >
                {/* Minimal header only (no raw head/tail list) */}
                {headerText ? (
                  <div style={{ marginBottom: '0.5rem', whiteSpace: 'pre-wrap' }}>
                    {headerText}
                  </div>
                ) : null}

                <ForecastSummary results={results} />

                <DataTable rows={head} title="Forecast (Head)" />
                <DataTable rows={tail} title="Forecast (Tail)" />
              </div>
            );
          }

          return (
            <div
              key={msg.id}
              className={`message ${msg.sender} ${msg.isError ? 'error' : ''}`}
              style={{ whiteSpace: 'pre-wrap' }}
            >
              {msg.text}
            </div>
          );
        })}

        {isLoading && (
          <div className="typing-indicator">
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
            <div className="typing-dot"></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="input-area" onSubmit={handleSend}>
        <input
          type="text"
          className="message-input"
          placeholder={datasetId ? "Type your message..." : "Upload a CSV first..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading || !datasetId}
        />
        <button
          type="submit"
          className="send-button"
          disabled={isLoading || !datasetId || !input.trim()}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
