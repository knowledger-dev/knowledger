// this is a custom hook that can be called so that we can use the dispatch and state context by using this hook

import { AuthContext } from "../context/AuthContext";
import { useContext } from "react"; // typically, you can just use this react hook, but we can make custom hooks for each context we have to make it better

export function useAuthContext() {
  const context = useContext(AuthContext); // this hook returns the value object when called

  if (!context) {
    throw Error("useAuthContext must be used inside a AuthContextProvider");
  }

  return context;
}
