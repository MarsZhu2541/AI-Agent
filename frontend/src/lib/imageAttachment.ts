import { SimpleImageAttachmentAdapter } from "@assistant-ui/react";

/** Inline images stay under DeepSeek's 32 MiB cap and the 48 MiB request body after base64. */
export const MAX_IMAGE_BYTES = 16 * 1024 * 1024;

const IMAGE_ACCEPT =
  "image/jpeg,image/jpg,image/png,image/gif,image/webp,.jpg,.jpeg,.png,.gif,.webp";

export class ImageAttachmentAdapter extends SimpleImageAttachmentAdapter {
  override accept = IMAGE_ACCEPT;

  override async add(state: { file: File }) {
    if (state.file.size > MAX_IMAGE_BYTES) {
      throw new Error(`图片不能超过 ${MAX_IMAGE_BYTES / (1024 * 1024)}MB`);
    }
    return super.add(state);
  }
}
