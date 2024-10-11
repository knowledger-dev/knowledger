import { useState } from "react";
import { useAuthContext } from "./useAuthContext";
import { toast } from 'react-toastify';
import * as CONSTANTS from "../BACKEND_VARS";

export const useLogin = () => {
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(null); // used to show when the api is loading
  const { dispatch } = useAuthContext(); // getting the dispatch function

  const login = async (username, password) => {
    setIsLoading(true);
    setError(null);
    toast.info('Logging in...', { autoClose: false });

    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const response = await fetch(`${CONSTANTS.BACKEND_HOST}/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" }, // we are sending form data
      body: formData.toString(), // converting form data to string
    });

    const json = await response.json();

    if (!response.ok) {
      setIsLoading(false);

      setError(json.error);

      toast.dismiss();  
      toast.error('Login failed. Please try again.');
      console.log("Error logging in:", json);
    }

    if (response.ok) {
      // save the user jwt to local storage
      localStorage.setItem("user", JSON.stringify(json)); // json is originally actually an object, so now we make it a json again

      // update the auth context

      dispatch({ type: "LOGIN", payload: json });

      toast.dismiss();
      toast.success('Login successful!');
      console.log("User logged in:", json);

      setIsLoading(false);
    }
  };

  return { login, isLoading, error };
};
