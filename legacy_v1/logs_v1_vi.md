Không cần sửa code, dùng shell để mỗi run tự lưu vào thư mục riêng, không đè lên nhau. `train.py` đã ghi `best_model.pth` + `training_history.json` (trong `checkpoints_optimized/`) và `test_results.json` (trong `results_optimized/`); mình chỉ archive chúng lại theo từng run + lưu cả log console.

Khi run đầu (đang chạy) xong, cất nó lại trước:
```
mkdir -p runs/run1
cp checkpoints_optimized/best_model.pth        runs/run1/
cp checkpoints_optimized/training_history.json runs/run1/
cp results_optimized/test_results.json         runs/run1/
cp config_optimized.yaml                        runs/run1/config.yaml
```

Mỗi run sau (nạp trọng số cũ, lưu đủ trọng số + lịch sử + log cho paper):
```
RUN=run2_$(date +%Y%m%d_%H%M)
mkdir -p runs/$RUN

python scripts/train.py --config config_optimized.yaml \
    --resume checkpoints_optimized/best_model.pth 2>&1 | tee runs/$RUN/train.log

cp checkpoints_optimized/best_model.pth        runs/$RUN/best_model.pth
cp checkpoints_optimized/training_history.json runs/$RUN/training_history.json
cp results_optimized/test_results.json         runs/$RUN/test_results.json
cp config_optimized.yaml                        runs/$RUN/config.yaml
```

Run sau nữa chỉ đổi `RUN=run3_...` rồi chạy lại y hệt. Mỗi thư mục `runs/<tên>/` giữ trọn: trọng số, lịch sử train từng epoch, kết quả test, config đúng lúc đó, và log console đầy đủ, đủ để dựng bảng/ђồ thị cho paper và truy vết.

Hai lưu ý:
- `tee` giữ toàn bộ output kể cả dòng log warm-start (bao nhiêu tensor full/partial/at-init), nên cho paper.
- `best_model.pth` là best của *chính run đó* (trainer reset bộ đếm best mỗi lần chạy). Muốn biết best xuyên suốt, so cột AUC trong các `test_results.json` giữa các run.