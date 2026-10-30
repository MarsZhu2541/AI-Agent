import { SimpleImageAttachmentAdapter } from "@assistant-ui/react";

/** DeepSeek accepts JPEG, PNG, GIF, and WebP, up to 8MB each. */
export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;

const IMAGE_ACCEPT =
  "image/jpeg,image/jpg,image/png,image/gif,image/webp,.jpg,.jpeg,.png,.gif,.webp";

export class ImageAttachmentAdapter extends SimpleImageAttachmentAdapter {
  override accept = IMAGE_ACCEPT;

  override async add(state: { file: File }) {
    if (state.file.size > MAX_IMAGE_BYTES) {
      throw new Error("图片不能超过 8MB");
    }
    return super.add(state);
  }
}
