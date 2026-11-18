// ==============================================================================
// DANH SÁCH ẢNH MẪU DÀNH CHO DEMO KIỂM THỬ NHANH
// ==============================================================================

window.SAMPLE_DATASETS = [
  {
    id: "sample_1",
    title: "Đường ngập có ô tô lưu thông",
    description: "Ảnh thực tế: nước phủ lòng đường đô thị, ô tô và người đi bộ di chuyển qua vùng ngập.",
    image: "samples/demo_urban_flood_traffic.jpg",
    expected: "Có dấu hiệu ngập lụt"
  },
  {
    id: "sample_2",
    title: "Ngập trước dãy cửa hàng",
    description: "Ảnh thực tế: nước tràn trên tuyến phố có cửa hàng và xe đỗ ven đường.",
    image: "samples/demo_urban_flood_shops.jpg",
    expected: "Có dấu hiệu ngập lụt"
  },
  {
    id: "sample_3",
    title: "Ô tô, xe máy qua đường ngập",
    description: "Ảnh thực tế: người dân và nhiều loại phương tiện cùng di chuyển trên đường ngập.",
    image: "samples/demo_urban_flood_motorbikes.jpg",
    expected: "Có dấu hiệu ngập lụt"
  },
  {
    id: "sample_4",
    title: "Đường khô — kiểm tra báo nhầm",
    description: "Ảnh đường phố có xe máy, không thấy vùng ngập; dùng để kiểm tra khả năng báo nhầm.",
    image: "samples/demo_urban_road_clear.jpg",
    expected: "Chưa phát hiện dấu hiệu ngập lụt"
  }
];
