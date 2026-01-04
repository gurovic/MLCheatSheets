#!/bin/bash

cd /home/runner/work/MLCheatSheets/MLCheatSheets/cheatsheets

# Get the HTML head/template from an existing file
HEAD=$(head -180 feature_importance_cheatsheet.html)
TAIL="</div>

</body>
</html>"

# Function to create a cheatsheet
create_cheatsheet() {
    local filename="$1"
    local title="$2"
    local emoji="$3"
    local subtitle="$4"
    local content="$5"
    
    cat > "$filename" << EOF
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>${title} — 3 колонки</title>
$(echo "$HEAD" | sed -n '4,179p')
</head>
<body>

<div class="container">

  <h1>${emoji} ${title}</h1>
  <div class="subtitle">${subtitle}<br>📅 Январь 2026</div>

${content}

</div>

</body>
</html>
EOF
    echo "Created: $filename ($(wc -c < "$filename") bytes)"
}

echo "Creating remaining 5 cheatsheets..."
create_cheatsheet "done.txt" "test" "🎯" "test subtitle" "<p>test</p>"
echo "Script ready"
