import { useRef, useEffect, useState } from "react";
import Bar from "../molecules/Bar";
import { AiOutlineSun, AiOutlineMoon } from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import Graph from "../molecules/Graph";
import "react-tabs/style/react-tabs.css";
import { toast } from "react-toastify";
// import * as CONSTANTS from "../BACKEND_VARS";
import NotePanel from "../molecules/NotePanel";

export default function Search() {
  const [focusedNode, setFocusedNode] = useState(null);
  const [isInputFocused, setIsInputFocused] = useState(false);
  const [data, setData] = useState({
    nodes: [],
    links: [],
  });
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [isBarOpen, setIsBarOpen] = useState(false);

  const [openNotes, setOpenNotes] = useState([]);
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const [isPaneVisible, setIsPaneVisible] = useState(false);

  const isDarkModeRef = useRef(
    localStorage.theme === "dark" || !("theme" in localStorage)
  );

  const [isDarkMode, setIsDarkMode] = useState(isDarkModeRef.current);

  useEffect(() => {
    if (isDarkModeRef.current) {
      document.documentElement.classList.add("dark");
      setIsDarkMode(true);
    } else {
      document.documentElement.classList.remove("dark");
      setIsDarkMode(false);
    }
  }, [isDarkModeRef]);

  function handleClick() {
    if (isDarkModeRef.current) {
      document.documentElement.classList.remove("dark");
      localStorage.theme = "light";
      setIsDarkMode(false);
    } else {
      document.documentElement.classList.add("dark");
      localStorage.theme = "dark";
      setIsDarkMode(true);
    }

    isDarkModeRef.current = !isDarkModeRef.current;
  }

  const handleChangeData = (newData) => {
    setData({
      nodes: newData.nodes,
      links: newData.links,
    });
  };

  const setClickOutsideListener = (handler) => {
    document.addEventListener("mousedown", handler);
  };

  const removeClickOutsideListener = (handler) => {
    document.removeEventListener("mousedown", handler);
  };

  const [currentMode, setCurrentMode] = useState(() => {
    const savedMode = localStorage.getItem("currentMode");
    return ["capture", "search", "searchai"].includes(savedMode)
      ? savedMode
      : "search"; // Default to "search" mode if not valid
  });

  useEffect(() => {
    localStorage.setItem("currentMode", currentMode);
  }, [currentMode]);

  const handleNodeClick = async (nodeName) => {
    // Find the full node object by its name
    const node = data.nodes.find((n) => n.name === nodeName);
    if (!node) {
      toast.error("Node not found.");
      return;
    }

    console.log("Node clicked:", node);

    // Check if note is already open
    const existingNoteIndex = openNotes.findIndex(
      (note) => note.id === node.id
    );

    if (existingNoteIndex !== -1) {
      setSelectedTabIndex(existingNoteIndex);
    } else {
      const noteContent = node.name;
      console.log("Adding note:", noteContent);
      setOpenNotes((prevNotes) => [...prevNotes, { noteContent, id: node.id }]);
      setSelectedTabIndex(openNotes.length); // Set the tab index to the newly added note
    }
    setIsPaneVisible(true); // Make pane visible when a note is clicked
  };

  return (
    <>
      <div className="fixed w-full z-0 inset-2">
        {data.nodes.length !== 0 && (
          <>
            <Graph
              data={data}
              isDarkMode={isDarkMode}
              focusedNode={focusedNode}
              setFocusedNode={setFocusedNode}
              setInfo={handleNodeClick}
              setIsPaletteOpen={setIsPaletteOpen}
              setIsBarOpen={setIsBarOpen}
              setIsInputFocused={setIsInputFocused}
              isPaneVisible={isPaneVisible}
            />
          </>
        )}
      </div>
      <IconButton onClick={handleClick} className="absolute right-0 p-5 top-0">
        {isDarkMode ? (
          <AiOutlineSun size={25} className="text-white" />
        ) : (
          <AiOutlineMoon size={25} className="text-white" />
        )}
      </IconButton>
      <section
        className={`font-inter font-semibold w-full fixed z-10 transition-transform duration-500`}
      >
        <div className="flex flex-col justify-center items-center z-10">
          <Bar
            isPaletteOpen={isPaletteOpen}
            setIsPaletteOpen={setIsPaletteOpen}
            isBarOpen={isBarOpen}
            setIsBarOpen={setIsBarOpen}
            setClickOutsideListener={setClickOutsideListener}
            removeClickOutsideListener={removeClickOutsideListener}
            currentMode={currentMode}
            setCurrentMode={setCurrentMode}
            isDarkMode={isDarkMode}
            handleChangeData={handleChangeData}
            setFocusedNode={setFocusedNode}
            setIsInputFocused={setIsInputFocused}
            isInputFocused={isInputFocused}
          />
        </div>
      </section>
      {isPaneVisible && (
        <NotePanel
          isPaneVisible={isPaneVisible}
          setIsPaneVisible={setIsPaneVisible}
          openNotes={openNotes}
          setOpenNotes={setOpenNotes}
          setSelectedTabIndex={setSelectedTabIndex}
          selectedTabIndex={selectedTabIndex}
        />
      )}
    </>
  );
}
