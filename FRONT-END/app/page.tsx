"use client";

import React, { useState, useRef, useEffect } from "react";
import { 
  Send, Upload, Bot, User, Terminal, Loader2, 
  AlertCircle, CheckCircle2, Info, Plus, 
  MessageSquare, Trash2, Menu, X 
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import WelcomeMessage from "@/components/WelcomeMessage";
import TypingIndicator from "@/components/TypingIndicator";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

//type Language = "fr" | "en" | "es" | "de";

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

interface Log {
  timestamp: string;
  level: "INFO" | "ERROR" | "WARNING" | "DEBUG";
  message: string;
  process_id: number;
  thread_id: number;
  memory_usage: number;
}

const LogIcon = ({ level }: { level: Log["level"] }) => {
  switch (level) {
    case "ERROR":
      return <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />;
    case "INFO":
      return <Info className="w-4 h-4 text-blue-400 shrink-0" />;
    case "WARNING":
      return <AlertCircle className="w-4 h-4 text-yellow-400 shrink-0" />;
    default:
      return <CheckCircle2 className="w-4 h-4 text-gray-400 shrink-0" />;
  }
};

export default function ChatInterface() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [logs, setLogs] = useState<Log[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Remove the models fetching and state management

  // Setup WebSocket connection
  useEffect(() => {
    const wsUrl = `${process.env.NEXT_PUBLIC_API_URL?.replace('http', 'ws')}/ws`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWebsocket(ws);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'typing':
          setIsTyping(true);
          setStreamingMessage("");
          break;
          
        case 'chunk':
          setIsTyping(false);
          setStreamingMessage(data.full_content);
          break;
          
        case 'complete':
          setIsTyping(false);
          setStreamingMessage("");
          
          // Add final message to conversation
          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            content: data.content,
            role: "assistant",
            timestamp: new Date(),
          };
          
          setConversations((prev) =>
            prev.map((conv) => {
              if (conv.id === currentConversationId) {
                return {
                  ...conv,
                  messages: [...conv.messages, assistantMessage],
                };
              }
              return conv;
            })
          );
          setIsLoading(false);
          break;
          
        case 'error':
          setIsTyping(false);
          setStreamingMessage("");
          setIsLoading(false);
          console.error('WebSocket error:', data.content);
          break;
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWebsocket(null);
    };
    
    return () => {
      ws.close();
    };
  }, [currentConversationId]);

  // Setup log event source for real-time logs
  useEffect(() => {
    const eventSource = new EventSource(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
    eventSource.onopen = () => setIsConnected(true);
    eventSource.onmessage = (event) => {
      const log = JSON.parse(event.data);
      setLogs((prev) => [...prev, log]);
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };
    eventSource.onerror = () => {
      setIsConnected(false);
      eventSource.close();
      setTimeout(() => window.location.reload(), 5000);
    };
    return () => eventSource.close();
  }, []);

  // Load conversations from localStorage
  useEffect(() => {
    const savedConversations = localStorage.getItem("conversations");
    if (savedConversations) {
      const parsed = JSON.parse(savedConversations);
      setConversations(
        parsed.map((conv: any) => ({
          ...conv,
          createdAt: new Date(conv.createdAt),
          messages: (conv.messages || []).map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
        }))
      );
    }
  }, []);

  // Save conversations to localStorage
  useEffect(() => {
    localStorage.setItem("conversations", JSON.stringify(conversations));
  }, [conversations]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations]);

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "New Conversation",
      messages: [],
      createdAt: new Date(),
    };
    setConversations((prev) => [newConversation, ...prev]);
    setCurrentConversationId(newConversation.id);
  };

  const deleteConversation = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setConversations((prev) => prev.filter((conv) => conv.id !== id));
    if (currentConversationId === id) {
      setCurrentConversationId(conversations[0]?.id || null);
    }
  };

  const getCurrentConversation = () => conversations.find((conv) => conv.id === currentConversationId);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !currentConversationId || !websocket) return;
  
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
      timestamp: new Date(),
    };
  
    setConversations((prev) =>
      prev.map((conv) => {
        if (conv.id === currentConversationId) {
          const isFirstMessage = conv.messages.length === 0;
  
          return {
            ...conv,
            title: isFirstMessage ? input : conv.title,
            messages: [...conv.messages, userMessage],
          };
        }
        return conv;
      })
    );
  
    const messageText = input;
    setInput("");
    setIsLoading(true);
  
    // Send message via WebSocket
    websocket.send(JSON.stringify({
      type: "chat",
      text: messageText,
      language: "fr"
    }));
  };
  
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/upload-training-data`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const userMessage: Message = {
          id: Date.now().toString(),
          content: `File "${file.name}" uploaded successfully!`,
          role: "assistant",
          timestamp: new Date(),
        };

        if (currentConversationId) {
          setConversations((prev) =>
            prev.map((conv) => {
              if (conv.id === currentConversationId) {
                return {
                  ...conv,
                  messages: [...conv.messages, userMessage],
                };
              }
              return conv;
            })
          );
        }
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return   (
    <div className="flex h-screen bg-gray-900">
      {/* Sidebar */}
     
 {/* Bouton pour contrôler la sidebar - maintenant visible sur tous les appareils */}
 <Button
        variant="ghost"
        size="icon"
        className={cn(
          "fixed top-4 z-50 transition-all duration-300 ease-in-out hover:bg-gray-800",
          isSidebarOpen ? "left-[250px]" : "left-4"
        )}
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? (
          <X className="w-6 h-6 text-gray-400" />
        ) : (
          <Menu className="w-6 h-6 text-gray-400 hover:bg-gray-300/10" />
        )}
      </Button>

      {/* Sidebar avec animation */}
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-[250px] transform transition-transform duration-300 ease-in-out",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="h-full border-r border-gray-800 flex flex-col bg-gray-900">
          <div className="p-4 border-b border-gray-800 mt-14">
            <Button 
              onClick={createNewConversation}
              className="w-full bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 flex items-center justify-center gap-2"
            >
              <Plus className="w-4 h-4" />
              New Assistant
            </Button>
          </div>
          
          <ScrollArea className="flex-1">
            <div className="space-y-2 p-4">
              {conversations.map((conversation, index) => (
                <div
                  key={conversation.id || index}
                  className={cn(
                    "group flex items-center gap-3 rounded-lg px-3 py-2 hover:bg-gray-800/50 cursor-pointer transition-colors duration-200",
                    currentConversationId === conversation.id && "bg-gray-800/50"
                  )}
                  onClick={() => setCurrentConversationId(conversation.id)}
                >
                  <MessageSquare 
                    className={cn(
                      "w-4 h-4",
                      currentConversationId === conversation.id ? "text-emerald-400" : "text-gray-400"
                    )}
                  />
                  <div className="flex-1 truncate text-sm text-gray-200">
                    {conversation.title}
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="w-6 h-6 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    onClick={(e) => deleteConversation(conversation.id, e)}
                  >
                    <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-400" />
                  </Button>
                </div>
              ))}
            </div>
          </ScrollArea>

          <div className="p-4 border-t border-gray-800">
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <div className={cn("w-2 h-2 rounded-full", isConnected ? "bg-green-500" : "bg-red-500")} />
              <span>{isConnected ? "Connected" : "Disconnected"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Overlay pour fermer la sidebar */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main Content avec marge dynamique */}
      <div 
        className={cn(
          "flex-1 flex flex-col transition-all duration-300 ease-in-out",
          isSidebarOpen ? "ml-[250px]" : "ml-0"
        )}
      >        {/* Header */}
        <div className="border-b border-gray-800 p-4">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-emerald-400 mx-12 my-[7px]"> Mtn Ai</h1>

           
 
          
          </div>
        </div>

        {/* Chat Area */}
        <ResizablePanelGroup direction="horizontal" className="flex-1">
          <ResizablePanel defaultSize={70} className="flex flex-col">
            <ScrollArea className="flex-1 p-4">
               {getCurrentConversation()?.messages.length === 0 ? (
    <WelcomeMessage />
  ) : (
              <div className="space-y-4">
                {getCurrentConversation()?.messages.map(message => (
                  <div key={message.id} className={cn("flex w-full", message.role === "user" ? "justify-end" : "justify-start")}>
                    <Card className={cn(
                      "max-w-[80%] p-3 animate-in fade-in slide-in-from-bottom-1",
                      message.role === "user" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "bg-gray-800/50 text-gray-200 border-gray-700"
                    )}>
                      <div className="flex items-start gap-2">
                        {message.role === "assistant" ? <Bot className="w-5 h-5 mt-1 shrink-0" /> : <User className="w-5 h-5 mt-1 shrink-0" />}
                        <div className="space-y-1">
                          {message.role === "assistant" ? (
                            <div className="text-sm prose prose-invert prose-sm max-w-none">
                              <ReactMarkdown 
                                remarkPlugins={[remarkGfm]}
                                rehypePlugins={[rehypeHighlight]}
                                components={{
                                  h1: (props) => <h1 className="text-lg font-bold text-emerald-400 mb-2" {...props} />,
                                  h2: (props) => <h2 className="text-md font-semibold text-emerald-300 mb-2" {...props} />,
                                  h3: (props) => <h3 className="text-sm font-semibold text-emerald-200 mb-1" {...props} />,
                                  p: (props) => <p className="text-gray-200 mb-2" {...props} />,
                                  ul: (props) => <ul className="list-disc list-inside text-gray-200 mb-2 space-y-1" {...props} />,
                                  ol: (props) => <ol className="list-decimal list-inside text-gray-200 mb-2 space-y-1" {...props} />,
                                  li: ({ ...props}) => <li className="text-gray-200" {...props} />,
                                  code: ({ className, children, ...props}) => {
                                    const match = /language-(\w+)/.exec(className || '');
                                    return match ? (
                                      <code className={cn(className, "block bg-gray-800 text-gray-200 p-2 rounded text-xs font-mono overflow-x-auto")} {...props}>
                                        {children}
                                      </code>
                                    ) : (
                                      <code className={cn(className, "bg-gray-700 text-emerald-300 px-1 py-0.5 rounded text-xs font-mono")} {...props}>
                                        {children}
                                      </code>
                                    );
                                  },
                                  pre: (props) => <pre className="bg-gray-800 p-2 rounded overflow-x-auto mb-2" {...props} />,
                                  strong: (props) => <strong className="font-semibold text-emerald-300" {...props} />,
                                  em: (props) => <em className="italic text-gray-300" {...props} />,
                                  blockquote: (props) => <blockquote className="border-l-4 border-emerald-500 pl-4 italic text-gray-300 mb-2" {...props} />,
                                }}
                              >
                                {message.content}
                              </ReactMarkdown>
                            </div>
                          ) : (
                            <p className="text-sm">{message.content}</p>
                          )}
                          <p className="text-xs text-gray-400">{message.timestamp.toLocaleTimeString()}</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                ))}
                
                {/* Streaming message display */}
                {streamingMessage && (
                  <div className="flex w-full justify-start">
                    <Card className="max-w-[80%] p-3 animate-in fade-in slide-in-from-bottom-1 bg-gray-800/50 text-gray-200 border-gray-700">
                      <div className="flex items-start gap-2">
                        <Bot className="w-5 h-5 mt-1 shrink-0" />
                        <div className="space-y-1">
                          <div className="text-sm prose prose-invert prose-sm max-w-none">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              rehypePlugins={[rehypeHighlight]}
                              components={{
                                h1: ({ ...props}) => <h1 className="text-lg font-bold text-emerald-400 mb-2" {...props} />,
                                h2: ({ ...props}) => <h2 className="text-md font-semibold text-emerald-300 mb-2" {...props} />,
                                h3: ({ ...props}) => <h3 className="text-sm font-semibold text-emerald-200 mb-1" {...props} />,
                                p: ({ ...props}) => <p className="text-gray-200 mb-2" {...props} />,
                                ul: ({ ...props}) => <ul className="list-disc list-inside text-gray-200 mb-2 space-y-1" {...props} />,
                                ol: ({ ...props}) => <ol className="list-decimal list-inside text-gray-200 mb-2 space-y-1" {...props} />,
                                li: ({ ...props}) => <li className="text-gray-200" {...props} />,
                                code: ({ className, children, ...props}) => {
                                  const match = /language-(\w+)/.exec(className || '');
                                  return match ? (
                                    <code className={cn(className, "block bg-gray-800 text-gray-200 p-2 rounded text-xs font-mono overflow-x-auto")} {...props}>
                                      {children}
                                    </code>
                                  ) : (
                                    <code className={cn(className, "bg-gray-700 text-emerald-300 px-1 py-0.5 rounded text-xs font-mono")} {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                                pre: (props) => <pre className="bg-gray-800 p-2 rounded overflow-x-auto mb-2" {...props} />,
                                strong: (props) => <strong className="font-semibold text-emerald-300" {...props} />,
                                em: (props) => <em className="italic text-gray-300" {...props} />,
                                blockquote: (props) => <blockquote className="border-l-4 border-emerald-500 pl-4 italic text-gray-300 mb-2" {...props} />,
                              }}
                            >
                              {streamingMessage}
                            </ReactMarkdown>
                          </div>
                          <p className="text-xs text-gray-400">En cours...</p>
                        </div>
                      </div>
                    </Card>
                  </div>
                )}
                
                {/* Typing indicator */}
                {isTyping && (
                  <div className="flex w-full justify-start">
                    <Card className="max-w-[80%] p-3 animate-in fade-in slide-in-from-bottom-1 bg-gray-800/50 text-gray-200 border-gray-700">
                      <div className="flex items-start gap-2">
                        <Bot className="w-5 h-5 mt-1 shrink-0" />
                        <TypingIndicator />
                      </div>
                    </Card>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div> )}
            </ScrollArea>

            {/* Input Area */}
            <div className="border-t border-gray-800 p-4">
              <form onSubmit={handleSubmit} className="flex gap-2">
                <input type="file" className="hidden" ref={fileInputRef} accept=".csv,.txt" onChange={handleFileUpload} />
                <Button
                  type="button"
                  variant="ghost"
                  className="text-gray-400 hover:text-emerald-400 hover:bg-gray-800"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-5 h-5" />
                </Button>
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={currentConversationId ? "Type your message..." : "Select or create a conversation to start"}
                  className="flex-1 bg-gray-800/50 border-gray-700 focus:border-emerald-500 text-gray-200"
                  disabled={isLoading || !currentConversationId}
                />
                <Button
                  type="submit"
                  disabled={isLoading || !currentConversationId}
                  className="bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                >
                  {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                </Button>
              </form>
            </div>
          </ResizablePanel>

          <ResizableHandle />

        <ResizablePanel defaultSize={30} className="flex flex-col">
          <div className="flex items-center justify-between border-b border-gray-800 p-4">
            <div className="flex items-center gap-2">
              <Terminal className="w-5 h-5 text-emerald-400" />
              <h2 className="text-sm font-semibold text-emerald-400">System Logs</h2>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>{logs.length} entries</span>
            </div>
          </div>
         {/**/} <ScrollArea className="flex-1">
            <div className="space-y-2 p-4">
              {logs.map((log, index) => (
                <div key={index} className="flex items-start gap-2 text-xs font-mono animate-in fade-in">
                  <LogIcon level={log.level} />
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className={cn("font-semibold", log.level === "ERROR" && "text-red-400", log.level === "INFO" && "text-blue-400", log.level === "WARNING" && "text-yellow-400", log.level === "DEBUG" && "text-gray-400")}>{log.level}</span>
                      <span className="text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-gray-300 break-all">{log.message}</p>
                    <div className="flex items-center gap-4 text-gray-500"><span>PID: {log.process_id}</span><span>Thread: {log.thread_id}</span><span>Memory: {log.memory_usage.toFixed(2)} MB</span></div>
                  </div>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </ScrollArea>

          <div className="border-t border-gray-800 p-2">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <div className="flex items-center gap-2">
                <div className={cn("w-2 h-2 rounded-full", isConnected ? "bg-green-500" : "bg-red-500")} />
                <span>{isConnected ? "Connected to logs" : "Reconnecting..."}</span>
              </div>
              <Button variant="ghost" size="sm" className="hover:text-emerald-400" onClick={() => setLogs([])}>Clear logs</Button>
            </div>
          </div>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
    </div>
  );
}

