/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    fontFamily: {
      inter: ["Inter", "mono"],
    },
    extend: {
      colors: {
        sealbrown: "rgb(98, 27, 0)",
        russianviolet: "rgb(47, 1, 71)",
        lavender: "rgb(197, 137, 232)",
        bittersweet: "rgb(255, 102, 99)",
      },
    },
  },
  plugins: [],
};
