import PropTypes from "prop-types";
import { useState } from "react";
export default function IconButton({ children, onClick, className }) {
  const [isClicked, setIsClicked] = useState(false);

  const handleClick = (e) => {
    setIsClicked(true);
    onClick(e);
    setTimeout(() => setIsClicked(false), 300); // Adjust the timeout duration as needed
  };

  return (
    <div className={`flex justify-center items-center ${className}`}>
      <button
        onClick={handleClick}
        className={`relative bg-purple-900 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-full transform transition-transform duration-200 hover:-translate-y-0.5`}
      >
        {children}
        <span
          className={`absolute -inset-0.5 rounded-full ring-4 ring-purple-700 transition-opacity duration-300 ${
            isClicked ? "opacity-100" : "opacity-0"
          }`}
        ></span>
      </button>
    </div>
  );
}

IconButton.propTypes = {
  children: PropTypes.node.isRequired,
  onClick: PropTypes.func.isRequired,
};
