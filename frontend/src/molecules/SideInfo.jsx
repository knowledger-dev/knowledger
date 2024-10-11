import PropTypes from "prop-types";
import parse from "html-react-parser";
import { ImCancelCircle } from "react-icons/im";

export default function SideInfo({ info, setInfo }) {
  const containsHTML = /<\/?[a-z][\s\S]*>/i.test(info);

  const clearInfo = () => {
    setInfo("");
  };

  const addTextBlackClass = (htmlString) => {
    if (typeof htmlString !== "string") {
      return null;
    }
    return parse(htmlString, {
      replace: (domNode) => {
        if (domNode.attribs) {
          domNode.attribs.class = domNode.attribs.class
            ? `${domNode.attribs.class} text-black dark:text-white`
            : "text-black dark:text-white";
        }
      },
    });
  };

  return (
    <>
      {info && (
        <div className="bg-gray-100 dark:bg-gray-900 text-white dark:text-gray-200 p-4 h-full shadow-lg m-4">
          <div className="overflow-y-auto max-h-96">
            <button
              onClick={clearInfo}
              className="float-right text-red-900 hover:text-red-700 dark:text-red-400 dark:hover:text-red-600"
            >
              <ImCancelCircle
                size={20}
                className="text-red-900 hover:text-red-700"
              />
            </button>
            {containsHTML ? (
              <div className="break-words">{addTextBlackClass(info)}</div>
            ) : (
              <p className="break-words">{info}</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}

SideInfo.propTypes = {
  info: PropTypes.string.isRequired,
  setInfo: PropTypes.func.isRequired,
};
