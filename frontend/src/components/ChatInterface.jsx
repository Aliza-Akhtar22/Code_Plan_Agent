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

  // New: bot message that can include a table preview
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

  const addUserMessage = (text) => {
    setMessages(prev => [
      ...prev,
      { id: Date.now() + Math.random(), text, sender: 'user' }
    ]);
  };

  // Helper: render a JSON rows array as a table
  const PreviewTable = ({ rows }) => {
    if (!rows || rows.length === 0) {
      return <div style={{ marginTop: '0.5rem' }}>No preview rows available.</div>;
    }

    // Build a stable column list across all rows
    const colSet = new Set();
    rows.forEach(r => Object.keys(r || {}).forEach(k => colSet.add(k)));
    const columns = Array.from(colSet);

    return (
      <div style={{ marginTop: '0.75rem', overflowX: 'auto' }}>
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
                      verticalAlign: 'top'
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

      // ✅ Use table-based preview message instead of JSON stringify
      addBotPreviewMessage({
        filename: data.filename || file.name,
        dataset_id: data.dataset_id,
        headRows: data.preview?.head || [],
      });

      // Add a normal instruction message after the table preview
      addBotMessage(
        `Now type a message like "hi" to get the plan and proposed ds/y, then reply "confirm".`
      );

    } catch (err) {
      console.error(err);
      const msg = err?.message || 'Failed to upload file.';
      setUploadError(msg);
      addBotMessage(`Upload error: ${msg}`, true);
    } finally {
      setIsLoading(false);
      // Allow re-uploading the same file by resetting input value
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

      // Prefer assistant_message, but ALWAYS append results when present
      let botMessageText =
        data.assistant_message ||
        data.reply ||
        data.message ||
        "";

      if (data.results) {
        botMessageText += `\n\nResults:\n${JSON.stringify(data.results, null, 2)}`;
      }

      if (data.error) {
        botMessageText += `\n\nError:\n${data.error}`;
      }

      if (!botMessageText.trim()) {
        botMessageText = JSON.stringify(data, null, 2);
      }

      addBotMessage(botMessageText, Boolean(data.error));

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
          // ✅ Special rendering for upload preview message with table
          if (msg.kind === 'upload_preview') {
            return (
              <div
                key={msg.id}
                className={`message bot`}
                style={{ whiteSpace: 'pre-wrap' }}
              >
                <div>
                  <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>
                    File uploaded successfully: {msg.filename}
                  </div>
                  <div style={{ marginBottom: '0.5rem' }}>
                    Dataset ID: {msg.dataset_id}
                  </div>
                  <div style={{ fontWeight: 600, marginTop: '0.75rem' }}>
                    Top 5 rows:
                  </div>
                  <PreviewTable rows={msg.headRows} />
                </div>
              </div>
            );
          }

          // Default rendering for regular messages
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
