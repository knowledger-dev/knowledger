import { useState } from "react";
import { useLogin } from "../hooks/useLogin";
import { useNavigate } from "react-router-dom";

const Login = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const { login, isLoading, error } = useLogin();

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("Logging in for:", username);

    await login(username, password);
  };

  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="flex items-center justify-center min-h-screen bg-transparent text-black dark:text-white">
      <form
        className="bg-transparent p-6 rounded-lg w-full max-w-sm"
        onSubmit={handleSubmit}
      >
        <h3 className="text-xl font-semibold mb-4 text-center">Log in</h3>
        <label className="block text-sm">Username:</label>
        <input
          className="w-full py-2 mb-4 border-b-2 border-0 bg-transparent outline-none"
          onChange={(e) => setUsername(e.target.value)}
          value={username}
        />
        <label className="block text-sm">Password:</label>
        <div className="relative w-full">
          <input
            className="w-full py-2 mb-4 border-b-2 border-0 bg-transparent outline-none"
            type={showPassword ? "text" : "password"}
            onChange={(e) => setPassword(e.target.value)}
            value={password}
          />
          <button
            type="button"
            className="absolute right-0 top-0 mt-2 mr-2"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>
        <button
          className="w-full p-2 bg-russianviolet text-white rounded-md hover:bg-russianviolet/90 disabled:opacity-50"
          disabled={isLoading}
        >
          Log in
        </button>
        <button
          onClick={() => navigate("/signup")}
          className="py-2 text-center w-full hover:text-gray-700 dark:hover:text-gray-300"
        >
          Don{`'`}t have an account yet? Sign up!
        </button>
        {error && <div className="mt-4 text-red-500 text-center">{error}</div>}
      </form>
    </div>
  );
};

export default Login;
