import { AiOutlineSetting } from "react-icons/ai";

export default function NavBar() {
  return (
    <nav className="flex flex-col justify-between items-center bg-purple-900 w-max p-6 font-inter font-bold z-10">
      <a
        href="/"
        className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800"
        cursor="pointer"
      >
        Agent
      </a>
      <a
        href="/settings"
        className="text-white py-2 no-underline hover:text-gray-300 active:text-gray-800"
        cursor="pointer"
      >
        <AiOutlineSetting size={25} />
      </a>
    </nav>
  );
}
