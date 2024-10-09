# Knowledger

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

# Quick Start

1. **Clone the repository:**

   ```sh
   git clone https://github.com/your-repo/knowledger.git
   cd knowledger
   ```

2. **Navigate to the `frontend` folder:**

   ```sh
   cd frontend
   ```

3. **Start the frontend development server:**

   - Open a terminal and run:
     ```sh
     npm run dev
     ```
   - Open another terminal and run:
     ```sh
     npm run electron
     ```

4. **Navigate to the `backend` folder:**

   ```sh
   cd ../backend
   ```

5. **Start the backend server:**

   - Open a terminal and run:
     ```sh
     uvicorn main:app --reload
     ```

6. **Setup environment variables:**

   - Create a `.env` file in the `backend` folder. (Ask us for the required variables later.)

7. **Configure frontend to connect to backend:**
   - In the `frontend/src` folder, locate the `VARS` file.
   - Change the `HOST` variable to the URL where your backend is being hosted.
