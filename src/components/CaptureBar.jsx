// CaptureBar.jsx

// importing react and other modules
import { useState, useEffect, useRef } from "react";
import { toast, ToastContainer } from "react-toastify";
import {
  AiOutlineArrowRight,
  AiOutlineUpload,
  AiOutlineSearch,
  AiOutlineSun,
  AiOutlineMoon,
  AiFillHome,
  AiFillInfoCircle,
} from "react-icons/ai";
import { GiBrain } from "react-icons/gi";
import "react-toastify/dist/ReactToastify.css";
import Mousetrap from "mousetrap";
import "mousetrap-global-bind"; // Import the global-bind functionality for Mousetrap

/**
 * Dropdown Component
 * Handles the display of Resurface and Discover sections.
 */

// Proptypes for dropdown component
import PropTypes from "prop-types";

const Dropdown = ({
  showDropdown,
  toggleDropdown,
  resurfaceItems,
  discoverItems,
}) => {
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        toggleDropdown(false);
      }
    };

    if (showDropdown) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showDropdown, toggleDropdown]);

  return (
    <div className="dropdown-container" ref={dropdownRef}>
      <button
        className="dropdown-btn"
        onClick={() => toggleDropdown(!showDropdown)}
        aria-expanded={showDropdown}
        aria-controls="suggestions-dropdown"
        aria-label="Toggle Suggestions"
      >
        Suggestions
        <span className={`arrow ${showDropdown ? "down" : "right"}`}>›</span>
      </button>

      {/* Suggestions Dropdown */}
      {showDropdown && (
        <div
          id="suggestions-dropdown"
          className="suggestions-dropdown"
          role="region"
          aria-label="Suggestions Dropdown"
        >
          {resurfaceItems.length === 0 && discoverItems.length === 0 ? (
            <p>No suggestions available at the moment.</p>
          ) : (
            <div className="main-grid">
              {/* Resurface Section */}
              {resurfaceItems.length > 0 && (
                <div className="info-section resurface-section">
                  <h2>Resurface</h2>
                  <div className="info-grid">
                    {resurfaceItems.map((item) => (
                      <button
                        key={item.id}
                        className="info-item clickable"
                        onClick={() => {
                          toast.info(`Clicked on ${item.title}`, {
                            position: "top-center",
                            autoClose: 3000,
                            hideProgressBar: true,
                          });
                        }}
                        aria-label={`Resurface: ${item.title}`}
                      >
                        <AiFillHome className="item-icon" size={24} />
                        <span className="item-label">{item.title}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Discoveries Section */}
              {discoverItems.length > 0 && (
                <div className="info-section discoveries-section">
                  <h2>Discover</h2>
                  <div className="info-grid">
                    {discoverItems.map((item) => (
                      <button
                        key={item.id}
                        className="info-item clickable"
                        onClick={() => {
                          toast.info(`Clicked on ${item.title}`, {
                            position: "top-center",
                            autoClose: 3000,
                            hideProgressBar: true,
                          });
                        }}
                        aria-label={`Discover: ${item.title}`}
                      >
                        <AiFillInfoCircle className="item-icon" size={24} />
                        <span className="item-label">{item.title}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

Dropdown.propTypes = {
  showDropdown: PropTypes.bool.isRequired,
  toggleDropdown: PropTypes.func.isRequired,
  resurfaceItems: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
    })
  ).isRequired,
  discoverItems: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
    })
  ).isRequired,
};

/**
 * CaptureBar Component
 * Main component containing the capture/search bar and suggestions dropdown.
 */
const CaptureBar = () => {
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [isCaptureMode, setIsCaptureMode] = useState(true); // Set to true by default
  const [isFocused, setIsFocused] = useState(false); // Track if chatbar is focused
  const [showDropdown, setShowDropdown] = useState(false); // Control dropdown visibility
  const [theme, setTheme] = useState("dark"); // Initialize to 'dark' theme
  const [resurfaceItems, setResurfaceItems] = useState([]); // Placeholder for Resurface items
  const [discoverItems, setDiscoverItems] = useState([]); // Placeholder for Discover items
  const inputRef = useRef(null); // Reference to the input field

  // Focus the input field when the hotkey is pressed
  useEffect(() => {
    Mousetrap.bindGlobal("ctrl+space", () => {
      if (inputRef.current) {
        inputRef.current.focus();
        setIsFocused(true); // Highlight the chatbar
      }
    });

    // Hotkey to toggle between Capture and Search modes
    Mousetrap.bindGlobal("ctrl+m", () => {
      setIsCaptureMode((prev) => !prev);
      toast.info(
        isCaptureMode ? "Switched to Search Mode" : "Switched to Capture Mode",
        {
          position: "top-center",
          autoClose: 3000,
          hideProgressBar: true,
        }
      );
    });

    return () => {
      // Unbind hotkeys when component is unmounted
      Mousetrap.unbind("ctrl+space");
      Mousetrap.unbind("ctrl+m");
    };
  }, [isCaptureMode]);

  // Deselect the chatbar when pressing Esc
  useEffect(() => {
    Mousetrap.bindGlobal("esc", () => {
      if (inputRef.current) {
        inputRef.current.blur();
        setIsFocused(false); // Remove the highlight
      }
    });

    // Unbind the hotkey when component is unmounted
    return () => {
      Mousetrap.unbind("esc");
    };
  }, []);

  // Apply the saved theme or default to 'dark' on component mount
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") || "dark";
    setTheme(savedTheme);
    document.body.setAttribute("data-theme", savedTheme);
  }, []);

  // Toggle between light and dark themes
  const toggleTheme = () => {
    const newTheme = theme === "dark" ? "light" : "dark";
    setTheme(newTheme);
    document.body.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme); // Persist the theme choice
  };

  const handleInputChange = (e) => {
    setInput(e.target.value);
    if (error) {
      setError(null); // Clear error while typing
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSubmit(); // Validate on submit
    }
  };

  const handleSubmit = () => {
    if (!input.trim()) {
      setError("Input cannot be empty!");
      return;
    } else if (input.length > 100) {
      setError("Input cannot exceed 100 characters!");
      return;
    }

    if (isCaptureMode) {
      toast.success("Note captured successfully!", {
        position: "top-center",
        autoClose: 3000,
        hideProgressBar: true,
      });
    } else {
      toast.info(`Searching for "${input}"...`, {
        position: "top-center",
        autoClose: 3000,
        hideProgressBar: true,
      });
    }
    setInput("");
  };

  // Toggle between Search and Capture mode
  const toggleMode = () => {
    setIsCaptureMode((prev) => !prev);
  };

  return (
    <div className="capture-container">
      {/* Toast Container for Notifications */}
      <ToastContainer />

      {/* Theme Toggle Button */}
      <div
        className="theme-toggle btn"
        onClick={toggleTheme}
        aria-label="Toggle Theme"
      >
        {theme === "dark" ? (
          <AiOutlineSun size={24} />
        ) : (
          <AiOutlineMoon size={24} />
        )}
      </div>

      {/* Header */}
      <h1 className="header-text">Hi, what do you want to know?</h1>

      {/* Capture Bar */}
      <div className={`capture-bar-wrapper ${isFocused ? "focused" : ""}`}>
        {/* Mode Switch Button */}
        <button
          className="btn mode-switch-btn"
          onClick={toggleMode}
          aria-label="Toggle Mode"
        >
          <span className="mode-tooltip">
            {isCaptureMode
              ? "Capture Mode - Click to switch to Search Mode"
              : "Search Mode - Click to switch to Capture Mode"}
          </span>
          {isCaptureMode ? (
            <GiBrain size={24} />
          ) : (
            <AiOutlineSearch size={24} />
          )}
        </button>

        {/* Input Field */}
        <input
          type="text"
          className="capture-input"
          placeholder={isCaptureMode ? "Capture your thoughts..." : "Search..."}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyPress}
          ref={inputRef} // Attach input ref for focus
          onFocus={() => setIsFocused(true)} // Set focus state on input focus
          onBlur={() => setIsFocused(false)} // Remove focus state on input blur
        />

        {/* Upload Button */}
        <button
          className="btn upload-btn"
          aria-label="Upload"
          onClick={() => {
            toast.info("Upload feature is not implemented yet.", {
              position: "top-center",
              autoClose: 3000,
              hideProgressBar: true,
            });
          }}
        >
          <AiOutlineUpload size={24} />
        </button>

        {/* Submit Button */}
        <button
          className="btn submit-btn"
          onClick={handleSubmit}
          aria-label="Submit"
        >
          <AiOutlineArrowRight size={24} />
        </button>
      </div>

      {/* Error Message */}
      {error && <p className="error-text">{error}</p>}

      {/* Dropdown Component */}
      <Dropdown
        showDropdown={showDropdown}
        toggleDropdown={setShowDropdown}
        resurfaceItems={resurfaceItems}
        discoverItems={discoverItems}
      />
    </div>
  );
};

export default CaptureBar;
