import { useState } from "react";
import { useSignup } from "../hooks/useSignup";
import { useNavigate } from "react-router-dom";

const Signup = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const { signup, error, isLoading } = useSignup();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("Signing up with:", username, email, password);

    await signup(username, email, password);
  };

  const [showPassword, setShowPassword] = useState(false);

  return (
    <div className="flex items-center justify-center min-h-screen bg-transparent text-black dark:text-white">
      <form
        className="bg-transparent p-6 rounded-lg w-full max-w-sm"
        onSubmit={handleSubmit}
      >
        <h3 className="text-xl font-semibold mb-4 text-center">Sign up</h3>
        <label className="block text-sm">Username:</label>
        <input
          className="w-full py-2 mb-4 border-b-2 border-0 bg-transparent outline-none"
          onChange={(e) => setUsername(e.target.value)}
          value={username}
        />
        <label className="block text-sm">Email:</label>
        <input
          type="email"
          onChange={(e) => setEmail(e.target.value)}
          value={email}
          className="w-full py-2 mb-4 border-b-2 border-0 bg-transparent outline-none"
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
          Sign up
        </button>
        <button
          onClick={() => navigate("/login")}
          className="py-2 text-center w-full hover:text-gray-700 dark:hover:text-gray-300"
        >
          Have an account already? Log in!
        </button>
        {error && <div className="mt-4 text-red-500 text-center">{error}</div>}
      </form>
    </div>
  );
};

export default Signup;
