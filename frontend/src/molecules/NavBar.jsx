import { useAuthContext } from "../hooks/useAuthContext";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLogout } from "../hooks/useLogout";
import { AiOutlineMenu, AiOutlineClose } from "react-icons/ai";

export default function NavBar() {
  const { user } = useAuthContext();
  const [isCollapsed, setIsCollapsed] = useState(false);

  const { logout } = useLogout();
  const navigate = useNavigate();

  const handleButtonClick = (path) => {
    navigate(path); // Programmatically navigate to another route
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <nav className="flex fixed flex-col justify-center items-center font-inter font-bold z-50 gap-6 top-0 m-4 w-[5%]">
      <button
        onClick={toggleCollapse}
        className="text-black outline-lavender dark:text-white no-underline hover:text-gray-800 active:text-gray-700 dark:hover:text-gray-300 dark:active:text-gray-400 text-center rounded-md cursor-pointer"
      >
        {isCollapsed ? (
          <AiOutlineMenu size={25} />
        ) : (
          <AiOutlineClose size={25} />
        )}
      </button>
      {!isCollapsed && (
        <>
          <button
            onClick={() => handleButtonClick("/")}
            className="dark:text-white hover:text-gray-900 active:text-gray-800 text-black py-2 no-underline dark:hover:text-gray-300 dark:active:text-gray-400 cursor-pointer text-center rounded-md p-4"
          >
            Agent
          </button>
          {user && (
            <button
              onClick={logout}
              className="dark:text-white hover:text-gray-900 active:text-gray-800 text-black py-2 no-underline dark:hover:text-gray-300 dark:active:text-gray-400 cursor-pointer text-center rounded-md p-4"
            >
              Logout
            </button>
          )}
          {/* <button
            onClick={() => handleButtonClick("/settings")}
            className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800 cursor-pointer"
          >
            <AiOutlineSetting size={25} />
          </button> */}
        </>
      )}
    </nav>
  );
}
