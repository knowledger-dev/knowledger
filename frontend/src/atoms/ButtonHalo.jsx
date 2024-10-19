import PropTypes from "prop-types";
import React from "react";

export default function ButtonHalo({ children, onChangeMode }) {
  const [isHaloVisible, setIsHaloVisible] = React.useState(false);

  const handleClick = () => {
    setIsHaloVisible(true);
    setTimeout(() => setIsHaloVisible(false), 200); // Halo effect duration
    onChangeMode();
  };

  return (
    <div className="flex justify-center items-center">
      <button
        onClick={handleClick}
        className={`relative bg-russianviolet hover:bg-russianviolet/60 text-white font-bold py-2 px-4 rounded-full transform transition-transform hover:-translate-y-0.5 max-md:text-sm`}
      >
        {children}
        {isHaloVisible && (
          <span className="absolute inset-0 rounded-full ring-4 ring-purple-500 animate-ping"></span>
        )}
      </button>
    </div>
  );
}

ButtonHalo.propTypes = {
  children: PropTypes.node.isRequired,
  onChangeMode: PropTypes.func.isRequired,
};
