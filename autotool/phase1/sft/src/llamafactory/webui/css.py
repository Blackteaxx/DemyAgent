

CSS = r"""
.duplicate-button {
  margin: auto !important;
  color: white !important;
  background: black !important;
  border-radius: 100vh !important;
}

.thinking-summary {
  padding: 8px !important;
}

.thinking-summary span {
  border-radius: 4px !important;
  padding: 4px !important;
  cursor: pointer !important;
  font-size: 14px !important;
  background: rgb(245, 245, 245) !important;
}

.dark .thinking-summary span {
  background: rgb(73, 73, 73) !important;
}

.thinking-container {
  border-left: 2px solid #a6a6a6 !important;
  padding-left: 10px !important;
  margin: 4px 0 !important;
}

.thinking-container p {
  color: #a6a6a6 !important;
}

.modal-box {
  position: fixed !important;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* center horizontally */
  max-width: 1000px;
  max-height: 750px;
  overflow-y: auto;
  background-color: var(--input-background-fill);
  flex-wrap: nowrap !important;
  border: 2px solid black !important;
  z-index: 1000;
  padding: 10px;
}

.dark .modal-box {
  border: 2px solid white !important;
}
"""
