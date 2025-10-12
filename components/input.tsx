"use client";

import { toast } from "sonner";

interface InputProps {
  input: string;
  setInput: (value: string) => void;
  isGeneratingResponse: boolean;
  onSubmit: () => void;
}

export function Input({
  input,
  setInput,
  isGeneratingResponse,
  onSubmit,
}: InputProps) {
  return (
    <textarea
      className="mb-12 w-full bg-transparent outline-none resize-none min-h-12 placeholder:text-zinc-400"
      placeholder="મને વચનામૃત વિશે પ્રશ્નો પૂછો!"
      value={input}
      // autoFocus
      onChange={(event) => {
        setInput(event.currentTarget.value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();

          if (input === "") return;

          if (isGeneratingResponse) {
            toast.error("Please wait for the model to finish its response!");
            return;
          }

          onSubmit();
        }
      }}
    />
  );
}
