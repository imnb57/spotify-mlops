# How to Showcase with Ngrok

To share this app with a friend, you need to expose **both** the Backend and Frontend, and tell the Frontend where to find the public Backend.

## Step 1: Expose Backend
1.  Open a terminal.
2.  Run:
    ```powershell
    ngrok http 8000
    ```
3.  Copy the `Forwarding` URL (e.g., `https://aaaa-123-45.ngrok-free.app`).
    *   *Note: Keep this terminal open.*

## Step 2: Configure Frontend
You need to restart the frontend so it knows to talk to the *public* backend URL, not `localhost`.

1.  Stop your current `npm run dev` terminal (Ctrl+C).
2.  Run the following command (replace the URL with the one you just copied):
    ```powershell
    # Windows PowerShell
    $env:VITE_API_URL="https://aaaa-123-45.ngrok-free.app"; npm run dev -- --host
    ```

## Step 3: Expose Frontend
1.  Open a **new** terminal.
2.  Run:
    ```powershell
    ngrok http 5173
    ```
    *(Check your running frontend terminal to verify the port is 5173. If it says 5174, use that).*

## Step 4: Share!
Copy the **Frontend** ngrok URL (from Step 3) and send it to your friend.

---

### ⚠️ Important Note on CORS
If your friend sees network errors, the Backend might be blocking the request.
We have already configured `CORSMiddleware` in `api/main.py` to allow `["*"]`, so it should work fine!
