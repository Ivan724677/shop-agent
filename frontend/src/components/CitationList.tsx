export default function CitationList({ ids = [] }: { ids?: string[] }) {
  if (!ids.length) return null;
  return (
    <div className="citation-list">
      <span>政策证据</span>
      {ids.map((id) => <code key={id}>{id}</code>)}
    </div>
  );
}

