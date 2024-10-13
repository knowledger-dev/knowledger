import PropTypes from "prop-types";
import { useEffect } from "react";
import { ImCancelCircle } from "react-icons/im";
import { Tab, Tabs, TabList, TabPanel } from "react-tabs";
import "react-tabs/style/react-tabs.css";

export default function NotePanel({
  isPaneVisible,
  setIsPaneVisible,
  openNotes,
  setOpenNotes,
  selectedTabIndex,
  setSelectedTabIndex,
}) {
  useEffect(() => {
    if (openNotes.length === 0) {
      setIsPaneVisible(false);
    }
  }, [openNotes, setIsPaneVisible]);

  if (!isPaneVisible) {
    return null;
  }

  return (
    <div className="fixed right-0 top-0 w-full transition-transform duration-500 translate-x-0">
      <section className="font-inter font-semibold fixed right-0 top-20 w-full max-w-[720px] h-[720px] z-10 bg-white dark:bg-gray-800/50 shadow-lg transition-transform duration-500 ease-in-out transform rounded-lg p-4 overflow-y-auto">
        <button
          aria-label="Close panel"
          onClick={() => setIsPaneVisible(false)}
          className="absolute right-4 top-4 m-2 p-2 bg-gray-300 dark:bg-gray-600 rounded-full hover:bg-gray-400 dark:hover:bg-gray-700"
        >
          <ImCancelCircle size={16} />
        </button>
        <Tabs
          selectedIndex={selectedTabIndex}
          onSelect={(index) => setSelectedTabIndex(index)}
        >
          <TabList className="bg-gray-200 dark:bg-gray-700 p-2 rounded-t-md flex overflow-x-auto w-full">
            {openNotes.map((note, index) => (
              <Tab
                key={note.id}
                className="px-4 py-2 m-1 p-2 border rounded-md cursor-pointer focus:outline-none hover:bg-gray-300 dark:hover:bg-gray-600 whitespace-nowrap flex justify-between gap-2"
              >
                {note.name || `Note ${index + 1}`}
                <button
                  aria-label={`Close ${note.name || `Note ${index + 1}`}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenNotes((prevNotes) =>
                      prevNotes.filter((n) => n.id !== note.id)
                    );
                    if (selectedTabIndex >= openNotes.length - 1) {
                      setSelectedTabIndex(Math.max(0, selectedTabIndex - 1));
                    }
                  }}
                  className="text-red-500 hover:text-red-700"
                >
                  <ImCancelCircle size={16} />
                </button>
              </Tab>
            ))}
          </TabList>
          <div className="p-2 dark:text-white text-black">
            {openNotes.map((note) => (
              <TabPanel key={note.id}>
                {note.id === "central" ? (
                  <div dangerouslySetInnerHTML={{ __html: note.noteContent }} />
                ) : (
                  note.noteContent
                )}
              </TabPanel>
            ))}
          </div>
        </Tabs>
      </section>
    </div>
  );
}

NotePanel.propTypes = {
  isPaneVisible: PropTypes.bool.isRequired,
  setIsPaneVisible: PropTypes.func.isRequired,
  openNotes: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      name: PropTypes.string,
      content: PropTypes.string,
    })
  ).isRequired,
  setOpenNotes: PropTypes.func.isRequired,
  selectedTabIndex: PropTypes.number.isRequired,
  setSelectedTabIndex: PropTypes.func.isRequired,
};
