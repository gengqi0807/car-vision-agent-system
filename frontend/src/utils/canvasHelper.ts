export interface Box {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

export function drawBoxes(ctx: CanvasRenderingContext2D, boxes: Box[]) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.strokeStyle = "#2dd4bf";
  ctx.lineWidth = 2;
  ctx.font = "14px sans-serif";
  boxes.forEach((box) => {
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillText(box.label, box.x, Math.max(18, box.y - 8));
  });
}
