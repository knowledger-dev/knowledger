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
      <section className="font-inter font-semibold fixed right-0 top-20 w-full max-w-[720px] max-h-[720px] z-10 bg-white dark:bg-gray-900/60 shadow-lg transition-transform duration-500 ease-in-out transform rounded-lg p-4">
        <button
          aria-label="Close panel"
          onClick={() => setIsPaneVisible(false)}
          className="absolute right-0 top-0 m-2 p-2 bg-transparent text-black dark:text-white rounded-full hover:text-gray-700 dark:hover:text-gray-300"
        >
          <ImCancelCircle size={16} />
        </button>
        <Tabs
          selectedIndex={selectedTabIndex}
          onSelect={(index) => setSelectedTabIndex(index)}
          selectedTabClassName="bg-lavender dark:bg-russianviolet text-black dark:text-white hover:bg-lavender dark:hover:bg-russianviolet"
        >
          <TabList className="border-b-2 m-0 pt-8 dark:border-white border-black flex w-full">
            {openNotes.map((note, index) => (
              <Tab
                key={note.id}
                className="px-4 p-2 border rounded-t-md cursor-pointer w-full text-black text-sm dark:text-white focus:outline-none hover:bg-gray-100 dark:hover:bg-gray-900 whitespace-nowrap flex justify-between gap-4"
              >
                {note.noteContent
                  ? `${note.noteContent.slice(
                      // need to fix, currently arbitrary, and does not work for central node
                      0,
                      openNotes.length > 5 ? 0 : 90 / openNotes.length ** 2
                    )}...`
                  : `Note ${index + 1}`}
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
                  className="text-white hover:text-red-200"
                >
                  <ImCancelCircle size={16} />
                </button>
              </Tab>
            ))}
          </TabList>
          <div className="p-2 dark:text-white text-black overflow-y-auto h-[600px]">
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
