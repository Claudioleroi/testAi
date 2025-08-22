import { Bot } from "lucide-react";
import { useEffect, useState } from "react";

const WelcomeMessage = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div className={`flex flex-col items-center justify-center h-full space-y-6 transition-opacity duration-1000 ${isVisible ? 'opacity-100' : 'opacity-0'}`}>
      <div className="relative">
        <div className="absolute -inset-1 bg-emerald-500/20 rounded-full blur animate-pulse" />
        <Bot className="w-16 h-16 text-emerald-400 relative" />
      </div>
      
      <div className="text-center space-y-4 max-w-lg">
        <h1 className="text-3xl font-bold text-emerald-400 animate-fade-in">
          Welcome to Mtn Ai Agent
        </h1>
        <p className="text-gray-400 text-lg animate-fade-in-delay">
          Your personal AI assistant where you can customize and train your own AI model
        </p>
        <div className="flex flex-wrap justify-center gap-3 text-sm text-gray-500 animate-fade-in-delay-2">
          <span className="px-3 py-1 bg-gray-800/50 rounded-full">Custom Training</span>
          <span className="px-3 py-1 bg-gray-800/50 rounded-full">Multiple Models</span>
          {/* <span className="px-3 py-1 bg-gray-800/50 rounded-full">Real-time Logs</span> */}
        </div>
      </div>
      
      <p className="text-gray-400 animate-bounce">
        Start a new conversation to begin
      </p>
    </div>
  );
};

export default WelcomeMessage;