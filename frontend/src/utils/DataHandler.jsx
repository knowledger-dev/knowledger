import { toast } from "react-toastify";
import * as CONSTANTS from "../BACKEND_VARS";

export async function AddSearch(search, user) {
  console.log("Adding search:", search);
  fetch(`${CONSTANTS.BACKEND_HOST}/notes`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${user.access_token}`, // sending the request with the user token
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      content: search,
      timestamp: new Date().toISOString(),
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      toast.success("Note saved successfully!");
      console.log("Note saved:", data);
    })
    .catch((error) => {
      console.error("Error:", error);
      toast.error("Failed to save note.", { autoClose: 5000 });
    });
}
