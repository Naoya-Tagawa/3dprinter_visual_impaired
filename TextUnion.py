from Bio import Align
from Bio.Seq import Seq


def merge_multiple_sequences(*sequences):
    aligner = Align.PairwiseAligner()

    # Biopython Seqオブジェクトに変換
    seq_objects = [Seq(seq) for seq in sequences]

    # 一つ目のシーケンスと他のシーケンスを順番にアラインメントしてマージ
    merged_seq = seq_objects[0]
    for seq in seq_objects[1:]:
        alignments = aligner.align(merged_seq, seq)
        if alignments:
            aligned_seq1, aligned_seq2 = alignments[0].aligned
            overlap_start = aligned_seq1[0]
            overlap_end = aligned_seq1[-1] + 1  # endはスライスに含まれないので+1
            merged_seq = merged_seq[:overlap_start] + seq[overlap_end:]
        else:
            # アラインメントが見つからなかった場合、単純に連結
            merged_seq += seq

    return str(merged_seq)


# テスト
text1 = "GGGGHHHko"
text2 = "kko.code"
text3 = "ko.code"
result = merge_multiple_sequences(text1, text2, text3)
print(result)
