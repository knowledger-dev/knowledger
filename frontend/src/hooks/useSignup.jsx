import { useState } from "react";
import { useAuthContext } from "./useAuthContext";
import { toast } from 'react-toastify';
import * as CONSTANTS from "../BACKEND_VARS";

export const useSignup = () => {
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(null); // used to show when the api is loading
  const { dispatch } = useAuthContext(); // getting the dispatch function

  const signup = async (username, email, password) => {
    setIsLoading(true);
    setError(null);

    const response = await fetch(`${CONSTANTS.BACKEND_HOST}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }, // we are sending json type data
      body: JSON.stringify({ username, email, password }), // making the body as a json
    });

    const json = await response.json();

    if (!response.ok) {
      setIsLoading(false);

      setError(json.error);
      toast.dismiss();  
      toast.error('Registration failed. Please try again.');
      console.log("Error signing up:", json);
    }

    if (response.ok) {
      // save the user jwt to local storage
      localStorage.setItem("user", JSON.stringify(json)); // json is originally actually an object, so now we make it a json again

      // update the auth context

      dispatch({ type: "LOGIN", payload: json });

      toast.dismiss();
      toast.success('Thank you for signing up! Login now to continue.');

      console.log("User signed up:", json);

      setIsLoading(false);
    }
  };

  return { signup, isLoading, error };
};
