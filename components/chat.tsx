"use client";

import cn from "classnames";
import { toast } from "sonner";
import { useChat } from "@ai-sdk/react";
import { useState } from "react";
import { Messages } from "./messages";
import { ArrowUpIcon, StopIcon } from "./icons";
import { Input } from "./input";

export function Chat() {
  const [input, setInput] = useState<string>("");

  const { messages, sendMessage, status, stop } = useChat({
    id: "primary",
    onError: () => {
      toast.error("An error occurred, please try again!");
    },
  });

  const isGeneratingResponse = ["streaming", "submitted"].includes(status);

  return (
    <div id="chat"
      className={cn(
        "px-4 md:px-0 pb-4 pt-8 flex flex-col h-dvh items-center w-full max-w-3xl",
        {
          "justify-between": messages.length > 0,
          "justify-center gap-4": messages.length === 0,
        }
      )}
    >
      {messages.length > 0 ? (
        <Messages messages={messages} status={status} />
      ) : (
        <div className="flex flex-col gap-0.5 sm:text-2xl text-xl w-full">
          <center>
          <b>
          <div className="dark:text-zinc-500 text-zinc-400">
            મને વચનામૃત વિશે પ્રશ્નો પૂછો!
          </div>
          </b>
          </center>
        </div>
      )}

      <div className="flex flex-col gap-4 w-full">
        <div className="flex relative flex-col gap-1 p-3 w-full rounded-2xl dark:bg-zinc-800 bg-zinc-100">
          <Input
            input={input}
            setInput={setInput}
            isGeneratingResponse={isGeneratingResponse}
            onSubmit={() => {
              if (input === "") return;
              sendMessage({
                role: "user",
                parts: [{ type: "text", text: input }],
              });
              setInput("");
            }}
          />

          <div className="absolute bottom-2.5 right-2.5 flex flex-row gap-2">
            <button
              className={cn(
                "size-8 flex flex-row justify-center items-center dark:bg-zinc-100 bg-zinc-900 dark:text-zinc-900 text-zinc-100 p-1.5 rounded-full hover:bg-zinc-800 dark:hover:bg-zinc-300 hover:scale-105 active:scale-95 transition-all",
                {
                  "dark:bg-zinc-200 dark:text-zinc-500":
                    isGeneratingResponse || input === "",
                }
              )}
              onClick={() => {
                if (input === "") return;
                if (isGeneratingResponse) {
                  stop();
                } else {
                  sendMessage({
                    role: "user",
                    parts: [{ type: "text", text: input }],
                  });
                }
                setInput("");
              }}
            >
              {isGeneratingResponse ? <StopIcon /> : <ArrowUpIcon />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
