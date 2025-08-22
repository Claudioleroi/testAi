import React from 'react';
import { cn } from '@/lib/utils';

interface TypingIndicatorProps {
  className?: string;
}

export default function TypingIndicator({ className }: TypingIndicatorProps) {
  return (
    <div className={cn("flex items-center space-x-1", className)}>
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
      </div>
      <span className="text-sm text-gray-400 ml-2">MTN AI is typing...</span>
    </div>
  );
}
