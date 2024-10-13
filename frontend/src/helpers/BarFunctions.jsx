export function CloseBar(setIsPaletteOpen, setIsInputFocused, setIsBarOpen) {
  setIsPaletteOpen(false);
  setIsInputFocused(false);
  setIsBarOpen(false);
}

export function handleClickOutside(
  e,
  setIsPaletteOpen,
  setIsInputFocused,
  setIsBarOpen
) {
  if (
    !e.target.closest("#chatbar-section") && // Ensure clicks on chatbar don't close the palette
    !e.target.closest("#command-palette") // Ensure clicks on command palette don't close the palette
  ) {
    CloseBar(setIsPaletteOpen, setIsInputFocused, setIsBarOpen);
  }
}
