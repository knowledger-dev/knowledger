import ButtonHalo from "../atoms/ButtonHalo";
import {
  AiOutlineArrowRight,
  AiOutlineSearch,
  AiOutlineUpload,
} from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import { useState } from "react";
import { GiBrain } from "react-icons/gi";

export default function Bar() {
  const [isCaptureMode, setIsCaptureMode] = useState(() => {
    const savedMode = localStorage.getItem("isCaptureMode");
    return savedMode !== null ? JSON.parse(savedMode) : false;
  });

  const onChangeMode = () => {
    const newMode = !isCaptureMode;
    setIsCaptureMode(newMode);
    localStorage.setItem("isCaptureMode", JSON.stringify(newMode));
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
    }
  };

  return (
    <>
      <section className="w-[40%] flex justify-center py-2 px-4 bg-slate-300 rounded-3xl dark:bg-gray-800 gap-2">
        <ButtonHalo onChangeMode={onChangeMode}>
          {/* Change icon of button on depending on Capture Mode */}
          {isCaptureMode ? (
            <GiBrain size={25} className="text-white" />
          ) : (
            <AiOutlineSearch size={25} className="text-white" />
          )}
        </ButtonHalo>

        {/* Change text input placeholder depending on Capture Mode */}
        {isCaptureMode ? (
          <input
            className="w-[80%] py-4 px-2 bg-slate-300 text-black dark:text-white dark:bg-gray-800 rounded-2xl text-lg focus:outline-none focus:shadow-outline"
            placeholder="Capture your thoughts..."
          />
        ) : (
          <input
            className="w-[80%] py-4 px-2 bg-slate-300 text-black dark:text-white dark:bg-gray-800 rounded-2xl text-lg focus:outline-none focus:shadow-outline"
            placeholder="Search..."
          />
        )}

        {/* Render different icons depending on Capture Mode */}
        {isCaptureMode ? (
          <>
            <IconButton
              onClick={() => document.getElementById("fileInput").click()}
            >
              <AiOutlineUpload size={25} className="text-white" />
            </IconButton>
            <input
              type="file"
              id="fileInput"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />
            <IconButton onClick={() => {}}>
              <AiOutlineArrowRight size={25} className="text-white" />
            </IconButton>
          </>
        ) : (
          <IconButton onClick={() => {}}>
            <AiOutlineArrowRight size={25} className="text-white" />
          </IconButton>
        )}
      </section>
    </>
  );
}
