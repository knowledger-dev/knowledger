import { useState } from "react";
import { AddSearch } from "../utils/DataHandler";
import { useAuthContext } from "../hooks/useAuthContext";

export default function CaptureTab() {
  const [search, setSearch] = useState("");
  const { user } = useAuthContext();

  const onSubmit = (e) => {
    e.preventDefault();
    AddSearch(search, user).then(() => {
      setSearch("");
    });
  };

  return (
    <>
      <form onSubmit={(e) => onSubmit(e)} className="w-full h-full">
        <textarea
          id="capture-tab"
          placeholder="Capture your thoughts here..."
          className="p-4 dark:bg-black dark:text-white text-black text-lg w-full h-full focus:outline-none"
          onChange={(e) => setSearch(e.target.value)}
          value={search}
        ></textarea>

        <button type="submit" className="absolute right-0 bottom-0 p-4">
          Add Note
        </button>
      </form>
    </>
  );
}
