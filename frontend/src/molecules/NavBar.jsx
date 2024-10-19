import { useAuthContext } from "../hooks/useAuthContext";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLogout } from "../hooks/useLogout";
import { AiOutlineMenu, AiOutlineClose } from "react-icons/ai";

export default function NavBar() {
  const { user } = useAuthContext();
  const [isCollapsed, setIsCollapsed] = useState(true);

  const { logout } = useLogout();
  const navigate = useNavigate();

  const handleButtonClick = (path) => {
    navigate(path); // Programmatically navigate to another route
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <>
      <button
        onClick={toggleCollapse}
        className=" z-50 outline-lavender p-5  no-underline text-center rounded-md cursor-pointer fixed top-0 left-0"
      >
        {isCollapsed ? (
          <div className="text-black dark:text-white">
            <AiOutlineMenu size={25} />
          </div>
        ) : (
          <div className="text-white">
            <AiOutlineClose size={25} />
          </div>
        )}
      </button>
      <nav
        className={`fixed bg-russianviolet flex flex-col font-inter font-bold gap-6 justify-start items-center top-0 left-0 h-full z-30 transition-transform duration-300 w-[250px] ${
          isCollapsed ? "-translate-x-full" : "translate-x-0"
        }`}
      >
        <button
          onClick={() => handleButtonClick("/")}
          className="dark:text-white  hover:text-gray-300 active:text-gray-400  text-white py-6 no-underline  cursor-pointer text-center rounded-md p-4"
        >
          Agent
        </button>
        {user && (
          <button
            onClick={logout}
            className="dark:text-white  hover:text-gray-300 active:text-gray-400  text-white py-6 no-underline  cursor-pointer text-center rounded-md p-4"
          >
            Logout
          </button>
        )}
      </nav>
    </>
  );
}
