// src/components/Bar.jsx
import { useCallback, useEffect, useState, useRef } from "react";
import ButtonHalo from "../atoms/ButtonHalo";
import {
  AiOutlineArrowRight,
  AiOutlineSearch,
  AiOutlineUpload,
} from "react-icons/ai";
import { LiaSearchPlusSolid } from "react-icons/lia";
import IconButton from "../atoms/IconButton";
import { GiBrain } from "react-icons/gi";
import { marked } from "marked";
import * as CONSTANTS from "../BACKEND_VARS";
import PropTypes from "prop-types";
import FocusableInput from "../atoms/FocusableInput";
import { toast } from "react-toastify";
import { useAuthContext } from "../hooks/useAuthContext";

const commands = [
  { id: "capture", label: "Switch to Capture Mode" },
  { id: "search", label: "Switch to Search Mode" },
  { id: "searchai", label: "Switch to AI Search Mode" },
  // Add more commands here if needed
];

export default function Bar({
  isDarkMode,
  handleChangeData,
  isInputFocused,
  setIsInputFocused,
  currentMode,
  setCurrentMode,
}) {
  const [search, setSearch] = useState("");
  const { user } = useAuthContext();
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [isBarOpen, setIsBarOpen] = useState(false);
  const [filteredCommands, setFilteredCommands] = useState(["No commands found."]);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const paletteRef = useRef(null);
  const inputRef = useRef(null); // Ref to manage FocusableInput

  // IPC Listener for 'focus-input' event
  useEffect(() => {
    const ipcRenderer = window.require("electron").ipcRenderer;

    const handleFocusInput = () => {
      setIsBarOpen((prev) => !prev);
      setSearch("");

      setIsPaletteOpen((prev) => !prev);
      setFilteredCommands(commands);
      setHighlightedIndex(-1);
      if (!isPaletteOpen && !isBarOpen) {
        setIsInputFocused(true);
        // Delay focus to ensure the input is rendered
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.focus();
          }
        }, 100);
      } else {
        setIsInputFocused(false);
      }
    };

    ipcRenderer.on("focus-input", handleFocusInput);

    return () => {
      ipcRenderer.removeListener("focus-input", handleFocusInput);
    };
  }, [isBarOpen, isPaletteOpen, setIsInputFocused]);

  // Close palette when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        paletteRef.current &&
        !paletteRef.current.contains(event.target) &&
        !event.target.closest("#chatbar-section") // Ensure clicks on chatbar don't close the palette
      ) {
        setIsPaletteOpen(false);
        setIsBarOpen(false);
        setIsInputFocused(false);
      }
    };

    if (isPaletteOpen && isBarOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isBarOpen, isPaletteOpen]);

  // Handle keyboard navigation within the command palette
  const handleKeyDown = useCallback(
    (e) => {
      if (isPaletteOpen && isBarOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setHighlightedIndex((prev) =>
            prev < filteredCommands.length - 1 ? prev + 1 : prev
          );
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : prev));
        } else if (e.key === "Enter") {
          e.preventDefault();
          if (
            highlightedIndex >= 0 &&
            highlightedIndex < filteredCommands.length
          ) {
            console.log("Doing a command.")
            console.log(highlightedIndex);
            executeCommand(filteredCommands[highlightedIndex].id);
          } else {
            // Default action if no command is highlighted
            console.log(highlightedIndex);
            console.log("Handling an action.")
            onSubmit(e);          }
          setSearch("");
        } else if (e.key === "Escape") {
          e.preventDefault();
          setIsPaletteOpen(false);
          setIsBarOpen(false);
          setIsInputFocused(false);
        }
      }
    },
    [isBarOpen, isPaletteOpen, highlightedIndex, filteredCommands]
  );

  // Add event listener for keydown when palette is open
  useEffect(() => {
    if (isPaletteOpen) {
      document.addEventListener("keydown", handleKeyDown);
      // Focus the input when palette opens
      if (inputRef.current) {
        inputRef.current.focus();
      }
    } else {
      document.removeEventListener("keydown", handleKeyDown);
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isPaletteOpen, handleKeyDown]);

  // Update filtered commands based on search input
  useEffect(() => {
    if (search.trim() === "") {
      setFilteredCommands(commands);
      setHighlightedIndex(-1);
    } else {
      const filtered = commands.filter((cmd) =>
        cmd.label.toLowerCase().includes(search.toLowerCase())
      );
      
      if (filtered.length === 0) {
        setFilteredCommands(["No commands found."]);
        setHighlightedIndex(-1);
      } else {
        setFilteredCommands(filtered);
        setHighlightedIndex(filtered.length > 0 ? 0 : -1);
      }
    }
  }, [search]);

  // Execute selected command
  const executeCommand = useCallback(
    (commandId) => {
      let newMode = commandId;
      if (newMode === currentMode) {
        toast.info(`Already in ${capitalize(newMode)} mode.`);
        return;
      }
      setCurrentMode(newMode);
      toast.info(`Switched to ${capitalize(newMode)} mode.`);
      localStorage.setItem("currentMode", newMode);
      setIsPaletteOpen(true);
      setIsBarOpen(true);
      setIsInputFocused(false);
      setSearch("");
    },
    [currentMode, setCurrentMode]
  );

  // Capitalize function for better readability in toasts
  const capitalize = (s) => {
    if (typeof s !== "string") return "";
    return s.charAt(0).toUpperCase() + s.slice(1);
  };

  // Function to handle mode switch from ButtonHalo
  const toggleNextMode = useCallback(() => {
    let newMode;
    if (currentMode === "capture") {
      newMode = "search";
      toast.info("Switched to Search Mode");
    } else if (currentMode === "search") {
      newMode = "searchai";
      toast.info("Switched to AI Query Mode");
    } else {
      newMode = "capture";
      toast.info("Switched to Capture Mode");
    }
    setCurrentMode(newMode);
    localStorage.setItem("currentMode", newMode);
  }, [currentMode, setCurrentMode]);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    toast.info("Uploading file...", { autoClose: false });
    if (file) {
      console.log(file);
      toast.success("File uploaded successfully!");
    }
  };

  // Function to calculate node value based on label size
  const calculateNodeValue = (label) => {
    // Placeholder function; adjust as needed
    console.log(label);
    return 2;
  };

  // Function to handle form submission
  const onSubmit = (event) => {
    event.preventDefault();
    console.log(`Search sent in ${currentMode} mode: `, search);

    if (search === "") {
      return;
    }

    if (isPaletteOpen) {
      setIsPaletteOpen(false);
      setIsBarOpen(false);
      setIsInputFocused(false);
    }

    if (currentMode === "capture") {
      toast.info("Saving note...", { autoClose: false });
      console.log("Saving note...");
      fetch(`${CONSTANTS.BACKEND_HOST}/notes`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: search,
          timestamp: new Date().toISOString(),
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          toast.success("Note saved successfully!");
          console.log("Note saved:", data);
          setSearch("");
        })
        .catch((error) => {
          console.error("Error:", error);
          toast.error("Failed to save note.", { autoClose: 5000 });
        });
    } else if (currentMode === "search") {
      toast.info("Searching...", { autoClose: false });
      setIsInputFocused(false);
      fetch(`${CONSTANTS.BACKEND_HOST}/query`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: search,
          limit: 10,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Search Basic Data: ", data);
          setSearch("");

          const nodes = data.map((parent) => ({
            id: parent._id,
            name: parent.content,
          }));

          const links = [];

          for (let i = 0; i < nodes.length; i++) {
            for (let j = 0; j < nodes.length; j++) {
              if (i !== j) {
                links.push({
                  source: nodes[i].id,
                  target: nodes[j].id,
                });
              }
            }
          }

          toast.success("Search loaded!");
          console.log("Nodes: ", nodes);
          console.log("Links: ", links);

          handleChangeData({ nodes, links });
        })
        .catch((error) => {
          toast.error("Error: " + error.message, { autoClose: false });
          console.error("Error:", error);
        });
    } else if (currentMode === "searchai") {
      toast.info("Knowledging...", { autoClose: false });
      setIsInputFocused(false);
      fetch(`${CONSTANTS.BACKEND_HOST}/rag_query`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: search,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          toast.dismiss();
          toast.success("Search loaded! Fetching notes...", { autoClose: false });
          console.log(data);
          setSearch("");

          const { answer, referenced_note_ids } = data;

          const fetchNotes = async (ids) => {
            const notes = await Promise.all(
              ids.map(async (id) => {
                const response = await fetch(
                  `${CONSTANTS.BACKEND_HOST}/notes/${id}`,
                  {
                    method: "GET",
                    headers: {
                      Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
                      "Content-Type": "application/json",
                    },
                  }
                );
                const note = await response.json();
                return { id: id, name: note.content };
              })
            );
            return notes;
          };

          fetchNotes(referenced_note_ids).then((notes) => {
            const centralNodeName = `<div style="font-size: 16px; color: #FFFFFF">${marked(
              answer
            ).replace(
              // Not currently working
              /\*\*(.*?)\*\*/g,
              "<h1 style='font-size: 20px'>$1</h1>"
            )}</div>`;

            handleChangeData({
              nodes: [
                {
                  id: "central",
                  name: centralNodeName,
                  val: 15,
                },
                ...notes.map((note) => ({
                  id: note.id,
                  name: note.name,
                  val: calculateNodeValue(note.name),
                })),
              ],
              links: notes.map((note) => ({
                source: "central",
                target: note.id,
              })),
            });
          });
        })
        .catch((error) => {
          toast.error("Error: " + error.message, {
            autoClose: false,
          });
          console.error("Error:", error);
        });
    }
  };

  return (
    <>
      {/* Chatbar Section */}
      {isBarOpen && (<section
        id="chatbar-section"
        className={`relative max-xl:w-[80%] w-[40%] flex justify-center py-2 px-4 bg-slate-300 dark:bg-gray-800 rounded-3xl gap-2 transition-[shadow,transform] duration-300 ${
          isInputFocused
            ? "shadow-lg shadow-purple-500/50 transform -translate-y-2"
            : ""
        }`}
      >
        {/* Mode Switcher Button */}
        <ButtonHalo onChangeMode={toggleNextMode}>
          {currentMode === "capture" ? (
            <GiBrain
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          ) : currentMode === "search" ? (
            <AiOutlineSearch
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          ) : (
            <LiaSearchPlusSolid
              size={25}
              className="text-white transition-transform duration-300"
            />
          )}
        </ButtonHalo>

        {/* Input Field */}
        <FocusableInput
          isDarkMode={isDarkMode}
          setIsInputFocused={setIsInputFocused}
          currentMode={currentMode}
          setSearch={setSearch}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isInputFocused && !isPaletteOpen) {
              onSubmit(e);
            }
          }}
          isInputFocused={isInputFocused}
          search={search}
          ref={inputRef} // Attach ref for focus management
        />

        {/* Conditional Icons based on Mode */}
        {currentMode === "capture" ? (
          <>
            <IconButton
              onClick={() => document.getElementById("fileInput").click()}
            >
              <AiOutlineUpload
                size={25}
                className="text-white transition-transform duration-300"
              />
            </IconButton>
            <input
              type="file"
              id="fileInput"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />
            <IconButton onClick={onSubmit}>
              <AiOutlineArrowRight
                size={25}
                className="text-white transition-transform duration-300"
              />
            </IconButton>
          </>
        ) : (
          <IconButton onClick={onSubmit}>
            <AiOutlineArrowRight
              size={25}
              className="text-white transition-transform duration-300"
            />
          </IconButton>
        )}
      </section>)}

      {/* Command Palette Dialog */}
      {isPaletteOpen && (
        <div
          ref={paletteRef}
          className={`absolute top-full mt-2 w-[40%] bg-slate-300 dark:bg-gray-800 rounded-2xl shadow-lg p-4 transition-opacity duration-300`}
        >
          <ul className="max-h-60 overflow-y-auto">
            {filteredCommands.map((cmd, index) => (
              <li
                key={cmd.id}
                className={`p-2 cursor-pointer rounded ${
                  highlightedIndex === index
                    ? "bg-gray-200 dark:bg-gray-700"
                    : "hover:bg-gray-100 dark:hover:bg-gray-600"
                }`}
                onMouseEnter={() => setHighlightedIndex(index)}
                onMouseLeave={() => setHighlightedIndex(-1)}
                onClick={() => executeCommand(cmd.id)}
              >
                {cmd.label}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Animation Keyframes (Tailwind CSS doesn't support custom keyframes out of the box) */}
      <style jsx>{`
        @keyframes fadeInUp {
          0% {
            opacity: 0;
            transform: translateY(-10px);
          }
          100% {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeInUp {
          animation: fadeInUp 0.3s ease-out forwards;
        }
      `}</style>
    </>
  );
}

Bar.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
  handleChangeData: PropTypes.func.isRequired,
  setFocusedNode: PropTypes.func.isRequired,
  isInputFocused: PropTypes.bool.isRequired,
  setIsInputFocused: PropTypes.func.isRequired,
  setCurrentMode: PropTypes.func.isRequired,
  currentMode: PropTypes.string.isRequired,
};
