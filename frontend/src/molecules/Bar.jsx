import { useCallback, useEffect, useState } from "react";
import Mousetrap from "mousetrap";
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

  const toggleNextMode = useCallback(() => {
    if (currentMode === "capture") {
      setCurrentMode("search");
      toast.info("Switched to Search Mode");
      localStorage.setItem("currentMode", "search");
    } else if (currentMode === "search") {
      setCurrentMode("searchai");
      toast.info("Switched to AI Query Mode");
      localStorage.setItem("currentMode", "searchai");
    } else {
      setCurrentMode("capture");
      toast.info("Switched to Capture Mode");
      localStorage.setItem("currentMode", "capture");
    }
  }, [currentMode, setCurrentMode]);

  // Function to toggle modes and save it to localStorage
  const onChangeMode = useCallback(() => {
    setIsInputFocused(false);
    const showCommandPalette = () => {
      const overlay = document.createElement("div");
      overlay.style.position = "fixed";
      overlay.style.top = 0;
      overlay.style.left = 0;
      overlay.style.width = "100%";
      overlay.style.height = "100%";
      overlay.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
      overlay.style.display = "flex";
      overlay.style.justifyContent = "center";
      overlay.style.alignItems = "center";
      overlay.style.zIndex = 1000;

      const palette = document.createElement("div");
      palette.style.backgroundColor = isDarkMode ? "#222" : "#fff";
      palette.style.padding = "20px";
      palette.style.borderRadius = "8px";
      palette.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
      palette.innerHTML = `
        <p>Press 'c' for Capture Mode</p>
        <p>Press 's' for Search Mode</p>
        <p>Press 'a' for SearchAI Mode</p>
      `;

      overlay.appendChild(palette);
      document.body.appendChild(overlay);

      const handleKeyPress = (event) => {
        const key = event.key.toLowerCase();
        let newMode;
        if (key === "c") {
          newMode = "capture";
          toast.info("Switched to Capture Mode");
        } else if (key === "s") {
          newMode = "search";
          toast.info("Switched to Search Mode");
        } else if (key === "a") {
          newMode = "searchai";
          toast.info("Switched to AI Query Mode");
        } else if (event.ctrlKey && event.shiftKey && key === "m") {
          document.body.removeChild(overlay);
          document.removeEventListener("keydown", handleKeyPress);
          setIsPaletteOpen(false);
        }

        if (newMode) {
          setCurrentMode(newMode);
          localStorage.setItem("currentMode", newMode);
          document.body.removeChild(overlay);
          document.removeEventListener("keydown", handleKeyPress);
          setIsPaletteOpen(false);
        }
      };

      document.addEventListener("keydown", handleKeyPress);
      setIsPaletteOpen(true);
    };

    if (!isPaletteOpen) {
      showCommandPalette();
    }
  }, [setCurrentMode, isPaletteOpen]);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    toast.info("Uploading file...", {autoClose: false});
    if (file) {
      console.log(file);
      toast.success("File uploaded successfully!");
    }
  };

  // Function to calculate node value based on label size
  const calculateNodeValue = (label) => {
    // const baseSize = 10; // You can adjust this base size as needed
    // return label.length / baseSize > 10 ? 10 : label.length / baseSize;
    console.log(label);
    return 2;
  };

  // useEffect to bind and unbind keyboard shortcuts
  useEffect(() => {
    Mousetrap.bindGlobal("ctrl+shift+m", onChangeMode);

    return () => {
      Mousetrap.unbind("ctrl+shift+m");
    };
  }, [currentMode, onChangeMode]);

  const onSubmit = (event) => {
    event.preventDefault();
    console.log(`Search sent in ${currentMode} mode: `, search);

    if (search === "") {
      return;
    }

    if (currentMode === "capture") {
      toast.info("Saving note...", {autoClose: false});
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
        });
    } else if (currentMode === "search") {
      toast.info("Searching...", {autoClose: false});
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

          toast.success("Search loaded!")
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
          toast.dismiss()
          toast.success("Search loaded! Fetching notes...", {autoClose: false})
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
            autoClose: false});
          console.error("Error:", error);
        });
    }
  };

  return (
    <>
      <section
        className={`max-xl:w-[80%] w-[40%] flex justify-center py-2 px-4 bg-slate-300 rounded-3xl dark:bg-gray-800 gap-2 transition-[shadow,transform] duration-300 ${
          isInputFocused
            ? "shadow-lg shadow-purple-500/50 transform -translate-y-2 transition-[shadow,transform] duration-300"
            : "transition-[shadow,transform] duration-300"
        }`}
      >
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
        <FocusableInput
          isDarkMode={isDarkMode}
          setIsInputFocused={setIsInputFocused}
          currentMode={currentMode}
          setSearch={setSearch}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isInputFocused) {
              onSubmit(e);
            }
          }}
          isInputFocused={isInputFocused}
          search={search}
        />
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
            <IconButton
              onClick={(e) => {
                onSubmit(e);
              }}
            >
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
      </section>
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
