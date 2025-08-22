// src/types/index.ts
interface Message {
    id: string;
    content: string;
    role: 'user' | 'assistant';
    timestamp: Date;
  }
  
  interface ChatConfig {
    companyName: string;
    welcomeMessage: string;
    primaryColor: string;
    logo?: string;
  }