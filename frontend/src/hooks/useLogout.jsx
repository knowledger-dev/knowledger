import { useAuthContext } from "./useAuthContext";
import { toast } from 'react-toastify';


export const useLogout = () => {
  const { dispatch } = useAuthContext();

  const logout = () => {
    // remove user from local storage
    localStorage.removeItem("user"); // we called it "user" when we made it

    toast.success('Logout successful!');
    // dispatch logout action
    dispatch({ type: "LOGOUT" });
  };

  return { logout };
};
