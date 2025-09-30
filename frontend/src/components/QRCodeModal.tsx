import { Modal } from 'antd'

interface QRCodeModalProps {
  title: string
  imageUrl: string
  modalText: string
  visible: boolean
  onClose: () => void
}

const QRCodeModal: React.FC<QRCodeModalProps> = ({ title, imageUrl, modalText, visible, onClose }) => {
  return (
    <Modal
      title={title}
      open={visible}
      onCancel={onClose}
      footer={null}
      width={300}
      centered
    >
      <div style={{ textAlign: 'center', padding: '20px 0' }}>
        <img src={imageUrl} alt={title} style={{ width: '100%', maxWidth: 250 }} />
        <p style={{ marginTop: 16, fontSize: 14, color: '#555' }}>
          {modalText}
        </p>
      </div>
    </Modal>
  )
}

export default QRCodeModal
