import { NextRequest } from "next/server";
import {
  createUIMessageStream,
  createUIMessageStreamResponse,
  UIMessage,
  convertToModelMessages,
} from "ai";

export const maxDuration = 30;

export async function POST(request: NextRequest) {
  try {
    const {
      messages,
      selectedModelId,
      isReasoningEnabled,
    }: {
      messages: Array<UIMessage>;
      selectedModelId?: string;
      isReasoningEnabled?: boolean;
    } = await request.json();

    const lastUser = [...(messages || [])]
      .reverse()
      .find((m) => m.role === "user");
    const query =
      lastUser?.parts
        ?.filter((p) => p.type === "text")
        .map((p) => ("text" in p ? p.text : ""))
        .join("") ?? "";

    const spaceRes = await fetch(process.env.HF_SPACE_URL as string, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    if (!spaceRes.ok) {
      const stream = createUIMessageStream({
        execute: async ({ writer }) => {
          writer.write({
            type: "data-notification",
            data: {
              message: `Space error: ${spaceRes.status} ${spaceRes.statusText}`,
              level: "error",
            },
            transient: true,
          });
          writer.write({ type: "text-start", id: "answer" });
          writer.write({
            type: "text-delta",
            id: "answer",
            delta: "An error occurred, please try again!",
          });
          writer.write({ type: "text-end", id: "answer" });
        },
      });
      return createUIMessageStreamResponse({ stream });
    }

    const data = (await spaceRes.json()) as {
      query?: string;
      answer?: string;
      selected?: unknown;
      expanded?: unknown;
    };

    const answer = (data?.answer ?? "").toString();
    const selected = data?.selected;
    const expanded = data?.expanded;

    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        if (selected !== undefined) {
          writer.write({
            type: "data-selected",
            data: selected,
          });
        }
        if (expanded !== undefined) {
          writer.write({
            type: "data-expanded",
            data: expanded,
          });
        }

        writer.write({ type: "text-start", id: "answer" });

        const encoder = new TextEncoder();

        const chunks = answer.split(/(\s+)/).filter(Boolean);
        for (const piece of chunks) {
          writer.write({ type: "text-delta", id: "answer", delta: piece });
        }

        writer.write({ type: "text-end", id: "answer" });
      },

      onError: (err) => `An error occurred, please try again!`,
    });

    return createUIMessageStreamResponse({
      stream,
    });
  } catch (error) {
    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        writer.write({ type: "text-start", id: "answer" });
        writer.write({
          type: "text-delta",
          id: "answer",
          delta: "An error occurred, please try again!",
        });
        writer.write({ type: "text-end", id: "answer" });
      },
    });
    return createUIMessageStreamResponse({ stream });
  }
}
