import { useState } from "react";
import { useLogin } from "../hooks/useLogin";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { login, isLoading, error } = useLogin();

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("Logging in with:", username, password);

    await login(username, password);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-transparent text-black">
      <form
        className="bg-white p-6 rounded-lg shadow-md w-full max-w-sm"
        onSubmit={handleSubmit}
      >
        <h3 className="text-xl font-semibold mb-4 text-center">Log in</h3>
        <label className="block mb-2">Username:</label>
        <input
          className="w-full p-2 mb-4 border rounded-md"
          onChange={(e) => setUsername(e.target.value)}
          value={username}
        />
        <label className="block mb-2">Password:</label>
        <input
          className="w-full p-2 mb-4 border rounded-md"
          type="password"
          onChange={(e) => setPassword(e.target.value)}
          value={password}
        />
        <button
          className="w-full p-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
          disabled={isLoading}
        >
          Log in
        </button>
        {error && <div className="mt-4 text-red-500 text-center">{error}</div>}
      </form>
    </div>
  );
};

export default Login;
