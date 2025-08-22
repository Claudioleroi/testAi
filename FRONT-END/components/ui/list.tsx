import React from 'react';

// Composant principal List
const List = ({ children }: { children: React.ReactNode }) => {
  return <ul className="space-y-2">{children}</ul>;
};

// Composant ListItem
const ListItem = ({ children }: { children: React.ReactNode }) => {
  return <li className="p-2 border rounded-lg">{children}</li>;
};

// Composant ListItemPrefix (Préfixe de l'élément de la liste)
const ListItemPrefix = ({ children }: { children: React.ReactNode }) => {
  return <span className="font-semibold text-gray-700">{children}</span>;
};

// Composant ListItemContent (Contenu de l'élément de la liste)
const ListItemContent = ({ children }: { children: React.ReactNode }) => {
  return <span className="ml-2 text-gray-600">{children}</span>;
};

export { List, ListItem, ListItemPrefix, ListItemContent };
