import { useAuthContext } from "../hooks/useAuthContext";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLogout } from "../hooks/useLogout";

export default function NavBar() {
  const { user } = useAuthContext();
  const [isLogin, setIsLogin] = useState(true);
  const [isNavVisible, setIsNavVisible] = useState(false);

  const { logout } = useLogout();

  const navigate = useNavigate();

  const handleButtonClick = (path) => {
    navigate(path); // Programmatically navigate to another route
  };

  const toggleLoginSignup = () => {
    setIsLogin(!isLogin);
    handleButtonClick(isLogin ? "/signup" : "/login");
  };

  const toggleNavVisibility = () => {
    setIsNavVisible(!isNavVisible);
  };

  return (
    <>
      <button
        onClick={toggleNavVisibility}
        className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800 cursor-pointer text-center rounded-md hover:bg-purple-400/55 p-4 fixed bottom-0 left-0 m-4 z-20 md:hidden"
      >
        {isNavVisible ? "Hide Menu" : "Show Menu"}
      </button>
      <nav
        className={`flex flex-col justify-start items-center bg-purple-900 w-max p-6 font-inter font-bold z-10 gap-6 transition-transform duration-300 ${
          isNavVisible ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0`}
      >
        <button
          onClick={() => handleButtonClick("/")}
          className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800 cursor-pointer text-center rounded-md hover:bg-purple-400/55 p-4"
        >
          Agent
        </button>
        {!user && (
          <button
            onClick={toggleLoginSignup}
            className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800 cursor-pointer text-center rounded-md hover:bg-purple-400/55 p-4"
          >
            {isLogin ? "Signup instead" : "Login instead"}
          </button>
        )}
        {user && (
          <button
            onClick={logout}
            className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800 cursor-pointer text-center rounded-md hover:bg-purple-400/55 p-4"
          >
            Logout
          </button>
        )}
      </nav>
    </>
  );
}
