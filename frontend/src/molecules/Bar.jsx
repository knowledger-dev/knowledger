import { useCallback, useEffect } from "react";
import Mousetrap from "mousetrap";
import ButtonHalo from "../atoms/ButtonHalo";
import {
  AiOutlineArrowRight,
  AiOutlineSearch,
  AiOutlineUpload,
} from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import { useState } from "react";
import { GiBrain } from "react-icons/gi";

import { marked } from "marked";

import BACKEND_HOST from "../../electron/VARS";

// Main functional component
import PropTypes from "prop-types";
import FocusableInput from "../atoms/FocusableInput";

export default function Bar({
  isDarkMode,
  handleChangeData,
  // setFocusedNode,
  isInputFocused,
  setIsInputFocused,
  isCaptureMode,
  setIsCaptureMode,
}) {
  const [search, setSearch] = useState("");

  // Function to toggle Capture Mode and save it to localStorage
  const onChangeMode = useCallback(() => {
    const newMode = !isCaptureMode;
    setIsCaptureMode(newMode);
    localStorage.setItem("isCaptureMode", JSON.stringify(newMode));
  }, [isCaptureMode, setIsCaptureMode]);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
    }
  };

  // Function to calculate node value based on label size
  const calculateNodeValue = (label) => {
    const baseSize = 10; // You can adjust this base size as needed
    return label.length / baseSize > 10 ? 10 : label.length / baseSize;
  };

  // useEffect to bind and unbind keyboard shortcuts
  useEffect(() => {
    Mousetrap.bindGlobal(["ctrl+shift+m", "command+shift+m"], onChangeMode);

    return () => {
      ~Mousetrap.unbind(["ctrl+shift+m", "command+shift+m"]);
    };
  }, [isCaptureMode, onChangeMode]);

  const onSubmit = (event) => {
    if (isCaptureMode) {
      event.preventDefault();
      console.log(search);

      if (search === "") {
        return;
      }

      if (search == "admin bypass populate") {
        fetch("./notes.txt")
          .then((response) => response.text())
          .then((text) => {
            const lines = text.split("\n");
            lines.forEach((line) => {
              fetch(`${BACKEND_HOST}/notes`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  content: line,
                  timestamp: new Date().toISOString(),
                }),
              })
                .then((response) => response.json())
                .then((data) => {
                  console.log("Note saved:", data);
                })
                .catch((error) => {
                  console.error("Error:", error);
                });
            });
          })
          .catch((error) => {
            console.error("Error reading file:", error);
          });
        return;
      }

      fetch(`${BACKEND_HOST}/notes`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: search,
          timestamp: new Date().toISOString(),
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Note saved:", data);
          setSearch("");
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } else {
      setIsInputFocused(false);
      setSearch("");
      fetch(`${BACKEND_HOST}/rag_query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: search,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data);

          const { answer, referenced_note_ids } = data;

          const fetchNotes = async (ids) => {
            const notes = await Promise.all(
              ids.map(async (id) => {
                const response = await fetch(`${BACKEND_HOST}/notes/${id}`, {
                  method: "GET",
                  headers: {
                    "Content-Type": "application/json",
                  },
                });
                const note = await response.json();
                return { id: note.id, name: note.content };
              })
            );
            return notes;
          };

          fetchNotes(referenced_note_ids).then((notes) => {
            const centralNodeName = `<div style="font-size: 16px; color: #FFFFFF; text-align: center">${marked(
              answer
            ).replace(
              // Not currently working
              /\*\*(.*?)\*\*/g,
              "<h1 style='font-size: 20px; text-align: center'>$1</h1>"
            )}</div>`;
            handleChangeData({
              nodes: [
                {
                  id: "central",
                  name: centralNodeName,
                  val: calculateNodeValue(answer > 100 ? 100 : answer),
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
        <ButtonHalo onChangeMode={onChangeMode}>
          {isCaptureMode ? (
            <GiBrain
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          ) : (
            <AiOutlineSearch
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          )}
        </ButtonHalo>
        {/* TODO: Currently, writing too much in the note goes past, and makes the placeholder look funky */}
        <FocusableInput
          isDarkMode={isDarkMode}
          setIsInputFocused={setIsInputFocused}
          isCaptureMode={isCaptureMode}
          setSearch={setSearch}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isInputFocused) {
              onSubmit(e);
            }
          }}
          isInputFocused={isInputFocused}
          search={search}
        />
        {isCaptureMode ? (
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
          <IconButton>
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
  setIsCaptureMode: PropTypes.func.isRequired,
  isCaptureMode: PropTypes.bool.isRequired,
};
