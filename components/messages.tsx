"use client";

import cn from "classnames";
import Markdown from "react-markdown";
import { markdownComponents } from "./markdown-components";
import { useEffect, useMemo, useRef } from "react";
import { UIMessage } from "ai";

function TextMessagePart({ text }: { text: string }) {
  return (
    <div className="flex flex-col gap-4">
      <Markdown components={markdownComponents}>{text}</Markdown>
    </div>
  );
}

export function Messages({
  messages,
  status,
}: {
  messages: Array<UIMessage>;
  status: "error" | "submitted" | "streaming" | "ready";
}) {
  const messagesRef = useRef<HTMLDivElement>(null);
  const messagesLength = useMemo(() => messages.length, [messages]);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messagesLength]);
  console.log("All messages:", messages);

  return (
    <div
      className="flex overflow-y-scroll flex-col gap-8 items-center w-full"
      ref={messagesRef}
    >
      {messages.map((message) => (
        <div
          key={message.id}
          className={cn(
            "flex flex-col gap-4 w-full last-of-type:mb-12 first-of-type:mt-16"
          )}
        >
          <div
            className={cn("flex flex-col gap-4", {
              "dark:bg-zinc-800 bg-zinc-200 p-2 rounded-xl w-fit ml-auto":
                message.role === "user",
              "": message.role === "assistant",
            })}
          >
           {message.parts.map((part, i) =>
  part.type === "text" ? (
    <TextMessagePart key={i} text={part.text} />
  ) : null
)}

          </div>
        </div>
      ))}

      {status === "submitted" && (
        <div className="mb-12 w-full text-zinc-500">Thinking...</div>
      )}
    </div>
  );
}
