async function init(){
  const respBoards = await fetch('/api/model/boards');
  const { boards } = await respBoards.json();

  const titleInput = document.getElementById('title');
  const predictBtn = document.getElementById('predictBtn');
  const resultDiv  = document.getElementById('result');

  predictBtn.addEventListener('click', () => { 
    predict(titleInput, boards, resultDiv) 
  });
};

async function predict(titleInput, boards, resultDiv){
  const title = titleInput.value.trim();
  if (!title) return alert('請先輸入標題');

  // 呼叫分類 API
  const resp = await fetch(`/api/model/prediction?title=${encodeURIComponent(title)}`);
  const { prediction } = await resp.json();
  
  // 先清空結果區
  resultDiv.textContent = "";

  // 1. 顯示預測結果
  const div = document.createElement("div");
  div.textContent = "預測結果：";
  div.append(" " + prediction);
  div.classList.add("bold-label");
  resultDiv.appendChild(div);

  // 2. 建立一個 flex 容器，放所有回饋按鈕
  const feedbackTitle = document.createElement("div");
  feedbackTitle.textContent = "選擇您認為更正確的分類";
  feedbackTitle.classList.add("big-label");
  resultDiv.appendChild(feedbackTitle);
  const btnContainer = document.createElement("div");
  btnContainer.classList.add("feedback-buttons");
  resultDiv.appendChild(btnContainer);

  // 3. 依照 boards 陣列，為每個板名建立一個按鈕
  boards.forEach(boardName => {
    const fbBtn = document.createElement("button");
    fbBtn.textContent = boardName;

    // 4. 直接在 click 時送出回饋
    fbBtn.addEventListener("click", async () => {
      try {
        const respFb = await fetch("/api/model/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: title,
            predicted_board: prediction,
            actual_board: boardName
          })
        });
        const { status } = await respFb.json();
        if (status === 'ok') {
          resultDiv.innerHTML = '<p>✅ 感謝您的回饋！</p>';
        } else {
          alert('回饋失敗，請再試一次');
        }
      } catch (err) {
        console.error(err);
        alert('網路錯誤，無法送出回饋');
      }
    });
    btnContainer.appendChild(fbBtn);
  });
}

init();