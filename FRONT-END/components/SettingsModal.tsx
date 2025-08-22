import React, { useState } from 'react';
import { Settings, Share2, Globe } from 'lucide-react';
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

type Language = 'fr' | 'en' | 'es' | 'de';

interface SettingsModalProps {
  currentLanguage: Language;
  onLanguageChange: (language: Language) => void;
  onShareChat: (chatId: string) => Promise<{ shareId: string }>;
  chatId: string | null;
}

const languages = [
  { code: 'fr' as Language, name: 'Français' },
  { code: 'en' as Language, name: 'English' },
  { code: 'es' as Language, name: 'Español' },
  { code: 'de' as Language, name: 'Deutsch' }
] as const;

const SettingsModal: React.FC<SettingsModalProps> = ({
  currentLanguage,
  onLanguageChange,
  onShareChat,
  chatId
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [shareUrl, setShareUrl] = useState('');

  const handleShare = async () => {
    if (!chatId) return;
    
    try {
      const response = await onShareChat(chatId);
      setShareUrl(`${window.location.origin}/shared-chat/${response.shareId}`);
    } catch (error) {
      console.error('Error sharing chat:', error);
    }
  };




  const handleLanguageChange = async (newLanguage: Language) => {
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/set-language`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language: newLanguage }),
      });
      onLanguageChange(newLanguage);  // Update local state
    } catch (error) {
      console.error('Error setting language:', error);
    }
  };



  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="text-gray-400 hover:text-emerald-400 hover:bg-gray-800"
        >
          <Settings className="w-5 h-5" />
        </Button>
      </DialogTrigger>
      <DialogContent className="bg-gray-900 border border-gray-800 text-gray-200">
        <DialogHeader>
          <DialogTitle className="text-emerald-400">Settings</DialogTitle>
          <DialogDescription className="text-gray-400">
            Configure your AI assistant preferences
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Language Selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-gray-200 flex items-center gap-2">
              <Globe className="w-4 h-4" />
              Interface Language
            </Label>
            <Select
              value={currentLanguage}
              onValueChange={(value) => handleLanguageChange(value as Language)}            >
              <SelectTrigger className="w-full bg-gray-800 border-gray-700">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent className="bg-gray-800 border-gray-700">
                {languages.map((lang) => (
                  <SelectItem
                    key={lang.code}
                    value={lang.code}
                    className="text-gray-200 hover:bg-gray-700"
                  >
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Share Chat */}
          <div className="space-y-2">
            <Label className="text-sm font-medium text-gray-200 flex items-center gap-2">
              <Share2 className="w-4 h-4" />
              Share Chat
            </Label>
            <div className="flex gap-2">
              <Input
                readOnly
                value={shareUrl}
                placeholder="Generate a share link"
                className="flex-1 bg-gray-800 border-gray-700 text-gray-200"
              />
              <Button
                onClick={handleShare}
                disabled={!chatId}
                className="bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 disabled:opacity-50"
              >
                Generate Link
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SettingsModal;