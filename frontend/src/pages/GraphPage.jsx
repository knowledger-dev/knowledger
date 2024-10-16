import { useEffect, useState } from "react";
import PropTypes from "prop-types";
import IconButton from "../atoms/IconButton";
import Graph from "../molecules/Graph";
import "react-tabs/style/react-tabs.css";
import { toast } from "react-toastify";
// import * as CONSTANTS from "../BACKEND_VARS";
import NotePanel from "../molecules/NotePanel";
import { useNavigate } from "react-router-dom";

export default function GraphPage({
  isDarkMode,
  data,
  currentMode,
  setIsPaletteOpen,
  setIsBarOpen,
  setIsInputFocused,
  focusedNode,
  setFocusedNode,
}) {
  const [openNotes, setOpenNotes] = useState([]);
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const [isPaneVisible, setIsPaneVisible] = useState(false);

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

  const navigate = useNavigate();

  const handleNavigateToCapture = () => {
    navigate("/capture");
  };

  return (
    <>
      <IconButton
        onClick={handleNavigateToCapture}
        className="p-3 z-50 absolute top-0 left-0"
      >
        Add Note
      </IconButton>
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

GraphPage.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
  data: PropTypes.object.isRequired,
  handleChangeData: PropTypes.func.isRequired,
  currentMode: PropTypes.string.isRequired,
  setIsPaletteOpen: PropTypes.func.isRequired,
  setIsBarOpen: PropTypes.func.isRequired,
  setIsInputFocused: PropTypes.func.isRequired,
  focusedNode: PropTypes.string,
  setFocusedNode: PropTypes.func.isRequired,
};
