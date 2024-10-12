import { useRef, useEffect, useState } from "react";
import Bar from "../molecules/Bar";
import { AiOutlineSun, AiOutlineMoon } from "react-icons/ai";
import { ImCancelCircle } from "react-icons/im";
import IconButton from "../atoms/IconButton";
import Graph from "../molecules/Graph";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";
import { toast } from "react-toastify";
import * as CONSTANTS from "../BACKEND_VARS";
import { useAuthContext } from "../hooks/useAuthContext";
import { marked } from 'marked';

export default function Search() {
  const [focusedNode, setFocusedNode] = useState(null);
  const [openNotes, setOpenNotes] = useState([]); // Array of open notes
  const [data, setData] = useState({
    nodes: [],
    links: [],
  });
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const { user } = useAuthContext();

  const isDarkModeRef = useRef(
    localStorage.theme === "dark" || !("theme" in localStorage)
  );

  const [isDarkMode, setIsDarkMode] = useState(isDarkModeRef.current);
  const [isPaneVisible, setIsPaneVisible] = useState(false);
  const [isPaneMinimized, setIsPaneMinimized] = useState(false);

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

  const [isInputFocused, setIsInputFocused] = useState(false);

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
    setIsPaneMinimized(false);

    // Find the full node object by its name
    const node = data.nodes.find((n) => n.name === nodeName);
    if (!node) {
      toast.error("Node not found.");
      return;
    }

    // Check if note is already open
    const existingNoteIndex = openNotes.findIndex((note) => note.id === node.id);

    if (existingNoteIndex !== -1) {
      setSelectedTabIndex(existingNoteIndex);
    } else {
      if (node.id !== "central") {
        try {
          const response = await fetch(
            `${CONSTANTS.BACKEND_HOST}/notes/${node.id}`,
            {
              method: "GET",
              headers: {
                Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
                "Content-Type": "application/json",
              },
            }
          );
          if (!response.ok) throw new Error("Note not found");
          const noteContent = await response.json();
          setOpenNotes((prevNotes) => [...prevNotes, { ...noteContent, id: node.id }]);
          setSelectedTabIndex(openNotes.length); // Set the tab index to the newly added note
        } catch (error) {
          toast.error("Failed to load note content.");
          console.error("Error fetching note:", error);
        }
      } else {
        // For Node.js
        var TurndownService = require('turndown')

        var turndownService = new TurndownService()
        var markdown = turndownService.turndown(nodeName)
        const noteContent = markdown; // turndownService.turndown(nodeName);
        setOpenNotes((prevNotes) => [...prevNotes, { content: noteContent, id: node.id }]);
        setSelectedTabIndex(openNotes.length); // Set the tab index to the newly added note
      }
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
        <div className="flex flex-col justify-center items-center">
          <Bar
            currentMode={currentMode}
            setCurrentMode={setCurrentMode}
            isDarkMode={isDarkMode}
            handleChangeData={handleChangeData}
            setFocusedNode={setFocusedNode}
            setIsInputFocused={setIsInputFocused}
            isInputFocused={isInputFocused}
          />
        </div>
        <div className="fixed right-0 top-0 w-[480px] h-[720px] z-10 transition-transform duration-500">
          <section
            className={`font-inter font-semibold fixed right-0 top-20 w-[720px] h-[720px] z-10 bg-white dark:bg-gray-800 shadow-lg transition-transform duration-500 ease-in-out transform ${isPaneVisible ? "translate-x-0" : "translate-x-full"}`}
            style={{ borderRadius: '12px', paddingBottom: '0.5rem' }}
          >
            {isPaneVisible && (
              <>
                <button style={{position: 'absolute', right: '0.5rem', top: '0.5rem'}} onClick={() => setIsPaneVisible(false)} className="m-2 p-2 bg-gray-300 dark:bg-gray-600 rounded">
                  <ImCancelCircle size={16} /> 
                </button>
                <Tabs selectedIndex={selectedTabIndex} onSelect={(index) => setSelectedTabIndex(index)}>
                  <TabList className="bg-gray-200 dark:bg-gray-700 p-2 rounded-t-md flex overflow-x-auto w-[100%]">
                    {openNotes.map((note, index) => (
                      <Tab
                        key={note.id}
                        className="relative px-4   py-2 m-1 border rounded-md cursor-pointer focus:outline-none hover:bg-gray-300 dark:hover:bg-gray-600 whitespace-nowrap"
                      >
                        {note.title || `Note ${index + 1}`}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenNotes((prevNotes) =>
                              prevNotes.filter((n) => n.id !== note.id)
                            );
                            if (selectedTabIndex >= openNotes.length - 1) {
                              setSelectedTabIndex(Math.max(0, selectedTabIndex - 1));
                            }
                          }}
                          style={{position: 'absolute', right: '0.5rem', top: '0.5rem'}}
                          className="mt-1 mr-1 text-red-500 hover:text-red-700"
                        >
                          <ImCancelCircle size={16} />
                        </button>
                      </Tab>
                    ))}
                  </TabList>
                  {openNotes.map((note) => (
                    <TabPanel key={note.id} className="px-6 py-2 overflow-y-auto max-h-[calc(100vh-21rem)]">
                      {note.content}
                    </TabPanel>
                  ))}
                </Tabs>
              </>
            )}
          </section>
        </div>
      </section>
    </>
  );
}